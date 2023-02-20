import pandas as pd
import yaml
from pathlib import Path


def load_entity_embeddings() -> pd.DataFrame:
    filepath = get_kg_path() / 'embeddings' / 'entity_embeddings.tsv'
    return pd.read_csv(filepath, sep='\t', header=None, index_col=0)


def load_entity_mapping() -> pd.DataFrame:
    filepath = get_kg_path() / 'entity_mapping.tsv'
    df = pd.read_csv(filepath, sep='\t', header=0)
    return df.dropna(subset=['source'])


def get_kg_result_path(run_id: str) -> Path:
    filepath = get_kg_path() / 'result' / f'run_{run_id}'
    filepath.mkdir(parents=True, exist_ok=True)
    return filepath


def load_config() -> dict:
    path_to_config = get_kg_path() / 'config.yaml'
    with open(path_to_config) as f:
        return yaml.safe_load(f)


def get_kg_path() -> Path:
    return Path('./kg')
