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


def make_embeddings(kg_config: dict, embedding_config: dict):
    get_logger().info('Starting rdf2vec embedding generation.')
    # create data in mapped-entity input format
    embedding_input_dir = Path(tempfile.mkdtemp())
    convert_graph_data(kg_config['format'], embedding_config['input_files'], embedding_input_dir, output_format='nt')
    # train embedding
    _train_embeddings(embedding_config, kg_config, embedding_input_dir)
    # serialize embedding
    _serialize_embeddings(embedding_config['models'], embedding_input_dir)


def _train_embeddings(embedding_config: dict, kg_config: dict, embedding_input_dir: Path):
    for model_name in embedding_config['models']:
        get_logger().info(f'Training embeddings of type {model_name}')
        command = [
            'conda', 'run', '--no-capture-output', '-n', 'jrdf2vec_env',
            'java', '-jar', '-Xmx300G', '/app/jrdf2vec.jar',
            '-graph', str(embedding_input_dir / 'train.nt'),
            '-walkGenerationMode', 'MID_WALKS_DUPLICATE_FREE',
            '-walkDirectory', str(embedding_input_dir / model_name),
            '-newFile', str(embedding_input_dir / f'{model_name}.txt'),
            '-threads', str(kg_config['max_cpus']),
            '-epochs', str(embedding_config['epochs'])
        ]
        get_logger().debug(f'Running command: {" ".join(command)}')
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        get_logger().debug(process.communicate()[1].decode())


def _serialize_embeddings(embedding_models: List[str], embedding_input_dir: Path):
    # load vectors of the respective models and merge indices with actual entity names
    entity_dict = pd.read_csv(embedding_input_dir / 'entities.dict', index_col=0, sep='\t', header=None, names=['entity'])
    for model_name in embedding_models:
        get_logger().info(f'Serializing embeddings of type {model_name}')
        embedding_file = embedding_input_dir / model_name / 'vectors.txt'
        embedding_vecs = pd.read_csv(embedding_file, sep=' ', index_col=0, header=None).dropna(how='all', axis='columns')
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
