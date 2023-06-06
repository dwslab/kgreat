import os
import yaml
import pandas as pd
from pathlib import Path


KG_DIR = Path('./kg')
DATA_DIR = KG_DIR / 'data'


def perform_identity_mapping(config: dict):
    path_to_entity_mapping = KG_DIR / 'entity_mapping.tsv'
    if not path_to_entity_mapping.is_file():
        raise FileNotFoundError('Could not find entity mapping file. Did you forget to `prepare` the mapping stage?')
    entities_to_map = pd.read_csv(path_to_entity_mapping, sep='\t', header=0)
    entities_to_map['source'] = entities_to_map[config['target']]
    if 'remove_prefix' in config:
        entities_to_map['source'] = entities_to_map['source'].str.removeprefix(config['remove_prefix'])
    if 'known_entity_file' in config:
        known_entities = set(pd.read_csv(DATA_DIR / config['known_entity_file'], sep='\t').iloc[:, 0])
        entities_to_map = entities_to_map[entities_to_map['source'].isin(known_entities)]
    entities_to_map.to_csv(path_to_entity_mapping, sep='\t', index=False)


if __name__ == "__main__":
    mapper_id = os.environ['KGREAT_STEP']
    with open(KG_DIR / 'config.yaml') as f:
        mapper_config = yaml.safe_load(f)['mapping'][mapper_id]
    perform_identity_mapping(mapper_config)
