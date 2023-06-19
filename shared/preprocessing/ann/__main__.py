from pathlib import Path
from logger import init_logger, get_logger
import yaml
import numpy as np
import pandas as pd
import hnswlib


KG_DIR = Path('./kg')
EMBEDDING_DIR = KG_DIR / 'embedding'


def make_ann_index(kg_config: dict):
    get_logger().info('Starting to build ANN indices')
    embedding_config = kg_config['preprocessing']['embedding']
    # build index for every embedding
    for model_name in embedding_config['models']:
        path_to_embedding_file = EMBEDDING_DIR / f'{model_name}.tsv'
        if not path_to_embedding_file.is_file():
            get_logger().info(f'Skipping ANN index computation for {model_name} as embedding vectors are not existing.')
            continue
        entity_vecs = pd.read_csv(path_to_embedding_file, sep='\t', header=None, index_col=0)
        get_logger().debug(f'Building ANN index for embedding {model_name}')
        index = _build_ann_index(kg_config, entity_vecs.values, 300, 32, 20)
        get_logger().debug(f'Persisting ANN index for embedding {model_name}')
        index.save_index(str(EMBEDDING_DIR / f'{model_name}_index.p'))


def _build_ann_index(kg_config: dict, embeddings: np.ndarray, ef_construction: int, M: int, ef: int) -> hnswlib.Index:
    index = hnswlib.Index(space='ip', dim=embeddings.shape[-1])
    index.init_index(max_elements=len(embeddings), ef_construction=ef_construction, M=M)
    index.add_items(embeddings, list(range(len(embeddings))), num_threads=kg_config['max_cpus'])
    index.set_ef(ef)
    return index


if __name__ == "__main__":
    with open(KG_DIR / 'config.yaml') as f:
        kg_config = yaml.safe_load(f)
    init_logger('preprocessing-ann', kg_config['log_level'])
    make_ann_index(kg_config)
