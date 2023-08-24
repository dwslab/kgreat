from typing import List
import os
import tempfile
from pathlib import Path
import yaml
import subprocess
import numpy as np
import pandas as pd
from logger import init_logger, get_logger
from converter import convert_graph_data


KG_DIR = Path('./kg')
EMBEDDING_DIR = KG_DIR / 'embedding'

EMBEDDING_BASE_CONFIGS = {
    'TransE_l1': ['--gamma', '12.0', '--lr', '0.007', '--regularization_coef', '2e-7'],
    'TransE_l2': ['--gamma', '10.0', '--lr', '0.1', '--regularization_coef', '1e-9'],
    'TransR': ['--gamma', '8.0', '--lr', '0.015', '--regularization_coef', '5e-8'],
    'DistMult': ['--gamma', '143.0', '--lr', '0.08'],
    'RESCAL': ['--gamma', '24.0', '--lr', '0.03'],
    'ComplEx': ['--gamma', '143.0', '--lr', '0.1', '--regularization_coef', '2e-6'],
}


def make_embeddings(kg_config: dict, embedding_config: dict):
    get_logger().info('Starting link-prediction embedding generation.')
    embedding_models = set(embedding_config['models'])
    # check if all specified embedding models are supported
    unsupported_models = embedding_models.difference(set(EMBEDDING_BASE_CONFIGS))
    if unsupported_models:
        get_logger().info(f'Skipping the following unsupported embedding models: {", ".join(unsupported_models)}')
        embedding_models = embedding_models.difference(unsupported_models)
    # check for existing embedding models
    existing_models = {m for m in embedding_models if (EMBEDDING_DIR / f'{m}.tsv').is_file()}
    if existing_models:
        get_logger().info(f'Skipping the following existing embedding models: {", ".join(existing_models)}')
        embedding_models = embedding_models.difference(existing_models)
    # finish early if no models to compute
    if not embedding_models:
        get_logger().info(f'No embeddings to compute. Finished.')
        return
    # create data in mapped-entity input format
    embedding_input_dir = Path(tempfile.mkdtemp())
    num_triples = convert_graph_data(kg_config['format'], embedding_config['input_files'], embedding_input_dir)
    # train embeddings
    _train_embeddings(embedding_config, kg_config, embedding_models, num_triples, embedding_input_dir)
    # serialize embeddings
    _serialize_embeddings(embedding_models, embedding_input_dir)


def _train_embeddings(embedding_config: dict, kg_config: dict, embedding_models: set, num_triples: int, embedding_input_dir: Path):
    get_logger().info(f'Training embeddings and storing at {embedding_input_dir}')
    batch_size = embedding_config['batch_size']
    neg_sample_size = 200
    max_steps = int(embedding_config['epochs']) * min(500000, num_triples * neg_sample_size // batch_size)

    for model_name in embedding_models:
        get_logger().info(f'Training embeddings of type {model_name}')
        command = [
            'dglke_train',
            '--model_name', model_name,
            '--dataset', 'kg',
            '--data_path', str(embedding_input_dir),
            '--save_path', str(embedding_input_dir),
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


def _serialize_embeddings(embedding_models: set, embedding_input_dir: Path):
    # load vectors of the respective models and merge indices with actual entity names
    entity_dict = pd.read_csv(embedding_input_dir / 'entities.dict', index_col=1, sep='\t', header=None, names=['entity'])
    for model_name in embedding_models:
        get_logger().info(f'Serializing embeddings of type {model_name}')
        embedding_folder = embedding_input_dir / '_'.join([model_name, 'kg', '0'])
        embedding_file = embedding_folder / '_'.join(['kg', model_name, 'entity.npy'])
        embedding_vecs = pd.DataFrame(data=np.load(str(embedding_file)), columns=range(200))
        entity_vecs = pd.merge(entity_dict, embedding_vecs, left_index=True, right_index=True).set_index('entity')
        get_logger().debug(f'Normalizing embeddings to unit vectors')
        entity_vecs = entity_vecs.apply(lambda x: x / np.linalg.norm(x, ord=1), axis=1).round(6)
        get_logger().debug(f'Writing embedding file')
        entity_vecs.to_csv(EMBEDDING_DIR / f'{model_name}.tsv', sep='\t', header=False)


if __name__ == "__main__":
    preprocessor_id = os.environ['KGREAT_STEP']
    with open(KG_DIR / 'config.yaml') as f:
        kg_config = yaml.safe_load(f)
    preprocessor_config = kg_config['preprocessing'][preprocessor_id]
    EMBEDDING_DIR.mkdir(exist_ok=True)
    init_logger(f'preprocessing-{preprocessor_id}', kg_config['log_level'], kg_config['run_id'])
    make_embeddings(kg_config, preprocessor_config)
