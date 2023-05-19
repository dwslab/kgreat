from typing import List, Optional
import datetime
from pathlib import Path
import logging
import yaml
import subprocess
import shutil
import numpy as np
import pandas as pd
from importer import get_reader_for_format


KG_DIR = Path('./kg')
DATA_DIR = KG_DIR / 'data'
EMBEDDINGS_DIR = KG_DIR / 'embeddings'

EMBEDDING_BASE_CONFIGS = {
    'TransE_l1': ['--gamma', '12.0', '--lr', '0.007', '--regularization_coef', '2e-7'],
    'TransE_l2': ['--gamma', '10.0', '--lr', '0.1', '--regularization_coef', '1e-9'],
    'TransR': ['--gamma', '8.0', '--lr', '0.015', '--regularization_coef', '5e-8'],
    'DistMult': ['--gamma', '143.0', '--lr', '0.08'],
    'RESCAL': ['--gamma', '24.0', '--lr', '0.03'],
    'ComplEx': ['--gamma', '143.0', '--lr', '0.1', '--regularization_coef', '2e-6'],
}


def make_embeddings(kg_config: dict):
    _get_logger().info('Starting embedding generation')
    # create data in dgl-ke input format
    _convert_graph_data(kg_config['format'])
    # check if all specified embedding models are supported
    embedding_config = kg_config['preprocessing']['embeddings']
    unsupported_models = set(embedding_config['models']).difference(set(EMBEDDING_BASE_CONFIGS))
    if unsupported_models:
        _get_logger().info(f'Skipping the following unsupported embedding models: {", ".join(unsupported_models)}')
        embedding_config['models'] = [m for m in embedding_config['models'] if m not in unsupported_models]
    # train and persist embeddings
    embedding_models = embedding_config['models']
    _cleanup_temp_embedding_folders(embedding_models)
    _train_embeddings(embedding_config, kg_config['gpu'])
    _serialize_embeddings(embedding_models)
    _cleanup_temp_embedding_folders(embedding_models)


def _convert_graph_data(kg_format: str):
    _get_logger().info(f'Converting input of format {kg_format} to dgl-ke format')
    reader = get_reader_for_format(kg_format)
    # gather entities, relations, triples
    entities, relations, triples = {}, {}, []
    for path in DATA_DIR.glob('*'):
        if not path.is_file():
            continue
        for s, r, o in reader.read(path):
            s_idx = _get_or_create_idx(s, entities)
            r_idx = _get_or_create_idx(r, relations)
            o_idx = _get_or_create_idx(o, entities)
            triples.append((s_idx, r_idx, o_idx))
    # persist in dgl-ke format
    _write_dglke_file([(idx, e) for e, idx in entities.items()], '\t', 'entities.dict')
    _write_dglke_file([(idx, r) for r, idx in relations.items()], '\t', 'relations.dict')
    _write_dglke_file(triples, '\t', 'train.tsv')


def _get_or_create_idx(value: str, value_dict: dict) -> str:
    if value not in value_dict:
        value_dict[value] = str(len(value_dict))
    return value_dict[value]


def _write_dglke_file(data: list, separator: str, filename: str):
    EMBEDDINGS_DIR.mkdir(exist_ok=True)
    filepath = EMBEDDINGS_DIR / filename
    with open(filepath, mode='w') as f:
        for vals in data:
            f.write(f'{separator.join(vals)}\n')


def _train_embeddings(embedding_config: dict, gpu: Optional[str]):
    for model_name in embedding_config['models']:
        _get_logger().info(f'Training embeddings of type {model_name}')
        command = [
            'dglke_train',
            '--model_name', model_name,
            '--dataset', 'kg',
            '--data_path', str(EMBEDDINGS_DIR),
            '--save_path', str(KG_DIR),
            '--data_files', 'entities.dict', 'relations.dict', 'train.tsv',
            '--format', 'udd_hrt',
            '--batch_size', str(embedding_config['batch_size']),
            '--neg_sample_size', '200',
            '--hidden_dim', '200',
            '--max_step', str(embedding_config['max_step']),
            '--log_interval', '1000',
            '-adv'
        ] + EMBEDDING_BASE_CONFIGS[model_name]
        if gpu != 'None':
            command += ['--gpu', gpu, '--mix_cpu_gpu']
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _get_logger().debug(process.communicate()[1].decode())


def _serialize_embeddings(embedding_models: List[str]):
    # load vectors of the respective models and merge indices with actual entity names
    entity_dict = pd.read_csv(EMBEDDINGS_DIR / 'entities.dict', index_col=0, sep='\t', header=None, names=['entity'])
    for model_name in embedding_models:
        _get_logger().info(f'Serializing embeddings of type {model_name}')
        embedding_folder = KG_DIR / '_'.join([model_name, 'kg', '0'])
        embedding_file = embedding_folder / '_'.join(['kg', model_name, 'entity.npy'])
        embedding_vecs = pd.DataFrame(data=np.load(str(embedding_file)), columns=range(200))
        entity_vecs = pd.merge(entity_dict, embedding_vecs, left_index=True, right_index=True)
        entity_vecs.to_csv(EMBEDDINGS_DIR / f'{model_name}.tsv', sep='\t', header=False, index=False)


def _cleanup_temp_embedding_folders(embedding_models: List[str]):
    for model_name in embedding_models:
        for dir in KG_DIR.glob(f'{model_name}_kg_*'):
            if dir.is_dir():
                shutil.rmtree(dir)


def _get_logger():
    return logging.getLogger('embeddings')


def _init_logger(log_level: str):
    log_filepath = EMBEDDINGS_DIR / '{}_embedding-generation.log'.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    log_handler = logging.FileHandler(log_filepath, 'a', 'utf-8')
    log_handler.setFormatter(logging.Formatter('%(asctime)s|%(levelname)s|%(module)s->%(funcName)s: %(message)s'))
    log_handler.setLevel(log_level)
    logger = _get_logger()
    logger.addHandler(log_handler)
    logger.setLevel(log_level)


if __name__ == "__main__":
    with open(KG_DIR / 'config.yaml') as f:
        kg_config = yaml.safe_load(f)
    _init_logger(kg_config['log_level'])
    make_embeddings(kg_config)
