from typing import Optional
import os
from pathlib import Path
from logger import init_logger, get_logger
import yaml
import numpy as np
import pandas as pd
import hnswlib


KG_DIR = Path('./kg')
EMBEDDING_DIR = KG_DIR / 'embedding'
EMBEDDING_SMALL_DIR = KG_DIR / 'embedding-small'
EMBEDDING_ANN_DIR = KG_DIR / 'embedding-ann'


def process_embeddings(kg_config: dict, speedup_config: dict):
    create_ann_index = 'ann_index' not in speedup_config or speedup_config['ann_index']
    create_small_embeddings = 'small_embeddings' not in speedup_config or speedup_config['small_embeddings']
    dataset_entities = _load_dataset_entities() if create_small_embeddings else None
    for embedding_file in EMBEDDING_DIR.glob('*.tsv'):
        model_name = embedding_file[:-4]
        get_logger().info(f'Making speed optimizations for embedding of type {model_name}')
        path_to_embedding_file = EMBEDDING_DIR / f'{model_name}.tsv'
        if not path_to_embedding_file.is_file():
            get_logger().info(f'Skipping optimizations as embedding vectors are not existing.')
            continue
        entity_vecs = pd.read_csv(path_to_embedding_file, sep='\t', header=None, index_col=0)
        if create_ann_index:
            get_logger().debug(f'Building ANN index for embedding {model_name}')
            index = _build_ann_index(kg_config, entity_vecs.values, 300, 32, 20)
            get_logger().debug(f'Persisting ANN index for embedding {model_name}')
            EMBEDDING_ANN_DIR.mkdir(exist_ok=True)
            index.save_index(str(EMBEDDING_ANN_DIR / f'{model_name}_index.p'))
        if dataset_entities:
            get_logger().debug(f'Persisting small embedding file for {model_name}')
            entity_vecs = entity_vecs[entity_vecs.index.isin(dataset_entities)]
            EMBEDDING_SMALL_DIR.mkdir(exist_ok=True)
            entity_vecs.to_csv(EMBEDDING_SMALL_DIR / f'{model_name}_small.tsv', sep='\t', header=False)


def _build_ann_index(kg_config: dict, embeddings: np.ndarray, ef_construction: int, M: int, ef: int) -> hnswlib.Index:
    index = hnswlib.Index(space='ip', dim=embeddings.shape[-1])
    index.init_index(max_elements=len(embeddings), ef_construction=ef_construction, M=M)
    index.add_items(embeddings, list(range(len(embeddings))), num_threads=kg_config['max_cpus'])
    index.set_ef(ef)
    return index


def _load_dataset_entities() -> Optional[set]:
    path_to_entity_mapping = KG_DIR / 'entity_mapping.tsv'
    if not path_to_entity_mapping.is_file():
        get_logger().info('Could not find entity mapping file - will skip creation of small embedding files.')
        return None
    dataset_entities = set(pd.read_csv(path_to_entity_mapping, sep='\t', header=0, dtype=str)['source'].values)
    dataset_entities = {ent for ent in dataset_entities if type(ent) == str and ent}
    if not dataset_entities:
        get_logger().info('Could not find any mapped entities in "entity_mapping.tsv" - will skip creation of small embedding files.')
        return None
    return dataset_entities


if __name__ == "__main__":
    preprocessor_id = os.environ['KGREAT_STEP']
    with open(KG_DIR / 'config.yaml') as f:
        kg_config = yaml.safe_load(f)
    preprocessor_config = kg_config['preprocessing'][preprocessor_id]
    init_logger(f'preprocessing-{preprocessor_id}', kg_config['log_level'], kg_config['run_id'])
    process_embeddings(kg_config, preprocessor_config)
