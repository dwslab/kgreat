from typing import List
import pandas as pd
import yaml
from pathlib import Path


def get_embedding_models() -> List[str]:
    return [embedding_file.stem for embedding_file in get_embedding_path().glob('*.tsv')]


def load_entity_embeddings(embedding_type: str, load_dataset_entities_only: bool) -> pd.DataFrame:
    full_embedding_path = get_embedding_path() / f'{embedding_type}.tsv'
    small_embedding_path = get_embedding_path(small=True) / f'{embedding_type}_small.tsv'
    use_small = load_dataset_entities_only and small_embedding_path.is_file()
    path_to_embeddings = small_embedding_path if use_small else full_embedding_path
    if not path_to_embeddings.is_file():
        raise FileNotFoundError(f'Trying to run task with {embedding_type} embeddings, but file is not existing.')
    return pd.read_csv(path_to_embeddings, sep='\t', header=None, index_col=0)


def load_entity_mapping() -> pd.DataFrame:
    filepath = get_kg_path() / 'entity_mapping.tsv'
    df = pd.read_csv(filepath, sep='\t', header=0, dtype=str)
    return df.dropna(subset=['source'])


def get_kg_result_path(run_id: str) -> Path:
    filepath = get_kg_path() / 'result' / f'run_{run_id}'
    filepath.mkdir(parents=True, exist_ok=True)
    return filepath


def load_kg_config() -> dict:
    path_to_config = get_kg_path() / 'config.yaml'
    with open(path_to_config) as f:
        return yaml.safe_load(f)


def load_dataset_config() -> dict:
    with open('config.yaml') as f:
        return yaml.safe_load(f)


def get_embedding_path(small: bool = False, ann: bool = False) -> Path:
    if small and ann:
        raise ValueError('Can only return folder for small embeddings OR ann indices.')
    postfix = ''
    if small:
        postfix = '-small'
    elif ann:
        postfix = '-ann'
    return get_kg_path() / f'embedding{postfix}'


def get_kg_path() -> Path:
    return Path('./kg')
