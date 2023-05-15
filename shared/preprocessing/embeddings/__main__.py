from typing import List
from pathlib import Path
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


def make_embeddings():
    # load and check KG config; create dgl-ke format
    with open(KG_DIR / 'config.yaml') as f:
        kg_config = yaml.safe_load(f)['kg']
    _convert_graph_data(kg_config)
    # check if all specified embedding models are supported
    embedding_config = kg_config['embeddings']
    unsupported_models = set(embedding_config['models']).difference(set(EMBEDDING_BASE_CONFIGS))
    if unsupported_models:
        print(f'Skipping the following unsupported embedding models: {", ".join(unsupported_models)}')  # TODO: logging!
        embedding_config['models'] = [m for m in embedding_config['models'] if m not in unsupported_models]
    # train and persist embeddings
    kg_name = kg_config['name']
    embedding_models = embedding_config['models']
    _cleanup_temp_embedding_folders(kg_name, embedding_models)
    _train_embeddings(kg_name, embedding_config)
    _serialize_embeddings(kg_name, embedding_models)
    _cleanup_temp_embedding_folders(kg_name, embedding_models)


def _convert_graph_data(kg_config: dict):
    reader = get_reader_for_format(kg_config['format'])
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


def _train_embeddings(kg_name: str, embedding_config: dict):
    for model_name in embedding_config['models']:
        command = [
            'dglke_train',
            '--model_name', model_name,
            '--dataset', kg_name,
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
        if 'gpu' in embedding_config and embedding_config['gpu'] != 'None':
            command += ['--gpu', str(embedding_config['gpu']), '--mix_cpu_gpu']
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(process.communicate()[1])  # TODO: Logging!
        print(f'Finished training model {model_name}')


def _serialize_embeddings(kg_name: str, embedding_models: List[str]):
    # load vectors of the respective models and merge indices with actual entity names
    entity_dict = pd.read_csv(EMBEDDINGS_DIR / 'entities.dict', index_col=0, sep='\t', header=None, names=['entity'])
    for model_name in embedding_models:
        embedding_folder = KG_DIR / '_'.join([model_name, kg_name, '0'])
        embedding_file = embedding_folder / '_'.join([kg_name, model_name, 'entity.npy'])
        embedding_vecs = pd.DataFrame(data=np.load(str(embedding_file)), columns=range(200))
        entity_vecs = pd.merge(entity_dict, embedding_vecs, left_index=True, right_index=True)
        entity_vecs.to_csv(EMBEDDINGS_DIR / f'{model_name}.tsv', sep='\t', header=False, index=False)


def _cleanup_temp_embedding_folders(kg_name: str, embedding_models: List[str]):
    for model_name in embedding_models:
        for dir in KG_DIR.glob(f'{model_name}_{kg_name}_*'):
            if dir.is_dir():
                shutil.rmtree(dir)


if __name__ == "__main__":
    make_embeddings()
