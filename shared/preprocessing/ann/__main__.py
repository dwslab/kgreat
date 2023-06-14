import datetime
from pathlib import Path
import logging
import yaml
import numpy as np
import pandas as pd
import hnswlib


KG_DIR = Path('./kg')
EMBEDDINGS_DIR = KG_DIR / 'embeddings'


def make_ann_index(kg_config: dict):
    _get_logger().info('Starting to build ANN indices')
    embedding_config = kg_config['preprocessing']['embeddings']
    # build index for every embedding
    for model_name in embedding_config['models']:
        path_to_embedding_file = EMBEDDINGS_DIR / f'{model_name}.tsv'
        if not path_to_embedding_file.is_file():
            _get_logger().info(f'Skipping ANN index computation for {model_name} as embedding vectors are not existing.')
            continue
        entity_vecs = pd.read_csv(path_to_embedding_file, sep='\t', header=None, index_col=0)
        _get_logger().debug(f'Building ANN index for embedding {model_name}')
        index = _build_ann_index(kg_config, entity_vecs.values, 300, 32, 20)
        _get_logger().debug(f'Persisting ANN index for embedding {model_name}')
        index.save_index(str(EMBEDDINGS_DIR / f'{model_name}_index.p'))


def _build_ann_index(kg_config: dict, embeddings: np.ndarray, ef_construction: int, M: int, ef: int) -> hnswlib.Index:
    index = hnswlib.Index(space='ip', dim=embeddings.shape[-1])
    index.init_index(max_elements=len(embeddings), ef_construction=ef_construction, M=M)
    index.add_items(embeddings, list(range(len(embeddings))), num_threads=kg_config['max_cpus'])
    index.set_ef(ef)
    return index


def _get_logger():
    return logging.getLogger('ann')


def _init_logger(log_level: str):
    log_filepath = EMBEDDINGS_DIR / '{}_ann-index.log'.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
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
    make_ann_index(kg_config)
