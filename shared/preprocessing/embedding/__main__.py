from typing import List
from pathlib import Path
import yaml
import subprocess
import shutil
import numpy as np
import pandas as pd
from logger import init_logger, get_logger
from importer import get_reader_for_format


KG_DIR = Path('./kg')
DATA_DIR = KG_DIR / 'data'
EMBEDDING_DIR = KG_DIR / 'embedding'

EMBEDDING_BASE_CONFIGS = {
    'TransE_l1': ['--gamma', '12.0', '--lr', '0.007', '--regularization_coef', '2e-7'],
    'TransE_l2': ['--gamma', '10.0', '--lr', '0.1', '--regularization_coef', '1e-9'],
    'TransR': ['--gamma', '8.0', '--lr', '0.015', '--regularization_coef', '5e-8'],
    'DistMult': ['--gamma', '143.0', '--lr', '0.08'],
    'RESCAL': ['--gamma', '24.0', '--lr', '0.03'],
    'ComplEx': ['--gamma', '143.0', '--lr', '0.1', '--regularization_coef', '2e-6'],
}


def make_embeddings(kg_config: dict):
    get_logger().info('Starting embedding generation')
    # check if all specified embedding models are supported
    embedding_config = kg_config['preprocessing']['embedding']
    unsupported_models = set(embedding_config['models']).difference(set(EMBEDDING_BASE_CONFIGS))
    if unsupported_models:
        get_logger().info(f'Skipping the following unsupported embedding models: {", ".join(unsupported_models)}')
        embedding_config['models'] = [m for m in embedding_config['models'] if m not in unsupported_models]
    # create data in dgl-ke input format
    num_triples = _convert_graph_data(kg_config['format'], embedding_config['input_files'])
    # train embeddings
    embedding_models = embedding_config['models']
    _cleanup_temp_embedding_folders(embedding_models)
    _train_embeddings(embedding_config, kg_config, num_triples)
    # serialize embeddings
    _serialize_embeddings_and_indices(embedding_models)
    _cleanup_temp_embedding_folders(embedding_models)


def _convert_graph_data(kg_format: str, input_files: List[str]) -> int:
    get_logger().info(f'Converting input of format {kg_format} to dgl-ke format')
    reader = get_reader_for_format(kg_format)
    # gather entities, relations, triples
    entities, relations, triples = {}, {}, []
    for filename in input_files:
        path = DATA_DIR / filename
        if not path.is_file():
            get_logger().info(f'Could not parse file "{filename}" as it is not contained in the data folder.')
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
    return len(triples)


def _get_or_create_idx(value: str, value_dict: dict) -> str:
    if value not in value_dict:
        value_dict[value] = str(len(value_dict))
    return value_dict[value]


def _write_dglke_file(data: list, separator: str, filename: str):
    filepath = EMBEDDING_DIR / filename
    with open(filepath, mode='w') as f:
        for vals in data:
            f.write(f'{separator.join(vals)}\n')


def _train_embeddings(embedding_config: dict, kg_config: dict, num_triples: int):
    batch_size = embedding_config['batch_size']
    neg_sample_size = 200
    max_steps = int(embedding_config['epochs']) * min(200000, num_triples * neg_sample_size // batch_size)

    for model_name in embedding_config['models']:
        get_logger().info(f'Training embeddings of type {model_name}')
        command = [
            'dglke_train',
            '--model_name', model_name,
            '--dataset', 'kg',
            '--data_path', str(EMBEDDING_DIR),
            '--save_path', str(KG_DIR),
            '--data_files', 'entities.dict', 'relations.dict', 'train.tsv',
            '--format', 'udd_hrt',
            '--batch_size', str(batch_size),
            '--neg_sample_size', str(neg_sample_size),
            '--hidden_dim', '200',
            '--max_step', str(max_steps),
            '--log_interval', '1000',
            '-adv'
        ] + EMBEDDING_BASE_CONFIGS[model_name]
        if kg_config['gpu'] != 'None':
            command += ['--gpu', kg_config['gpu'], '--mix_cpu_gpu']
        get_logger().debug(f'Running command: {" ".join(command)}')
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        get_logger().debug(process.communicate()[1].decode())


def _serialize_embeddings_and_indices(embedding_models: List[str]):
    # load vectors of the respective models and merge indices with actual entity names
    entity_dict = pd.read_csv(EMBEDDING_DIR / 'entities.dict', index_col=0, sep='\t', header=None, names=['entity'])
    for model_name in embedding_models:
        get_logger().info(f'Serializing embeddings of type {model_name}')
        embedding_folder = KG_DIR / '_'.join([model_name, 'kg', '0'])
        embedding_file = embedding_folder / '_'.join(['kg', model_name, 'entity.npy'])
        embedding_vecs = pd.DataFrame(data=np.load(str(embedding_file)), columns=range(200))
        entity_vecs = pd.merge(entity_dict, embedding_vecs, left_index=True, right_index=True).set_index('entity')
        get_logger().debug(f'Normalizing embeddings to unit vectors')
        entity_vecs = entity_vecs.apply(lambda x: x / np.linalg.norm(x, ord=1), axis=1).round(6)
        get_logger().debug(f'Writing embedding file')
        entity_vecs.to_csv(EMBEDDING_DIR / f'{model_name}.tsv', sep='\t', header=False)


def _cleanup_temp_embedding_folders(embedding_models: List[str]):
    for model_name in embedding_models:
        for dir in KG_DIR.glob(f'{model_name}_kg_*'):
            if dir.is_dir():
                shutil.rmtree(dir)


if __name__ == "__main__":
    with open(KG_DIR / 'config.yaml') as f:
        kg_config = yaml.safe_load(f)
    EMBEDDING_DIR.mkdir(exist_ok=True)
    init_logger('preprocessing-embedding', kg_config['log_level'])
    make_embeddings(kg_config)
