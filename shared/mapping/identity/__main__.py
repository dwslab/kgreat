import os
import yaml
import pandas as pd
from logger import init_logger, get_logger
from pathlib import Path
from importer import get_reader_for_format


KG_DIR = Path('./kg')
DATA_DIR = KG_DIR / 'data'


def perform_identity_mapping(kg_config: dict, mapper_config: dict):
    path_to_entity_mapping = KG_DIR / 'entity_mapping.tsv'
    if not path_to_entity_mapping.is_file():
        raise FileNotFoundError('Could not find entity mapping file. Did you forget to `prepare` the mapping stage?')
    entities_to_map = pd.read_csv(path_to_entity_mapping, sep='\t', header=0, dtype=str)
    entities_to_map['source'] = entities_to_map[mapper_config['target']]
    if 'remove_prefix' in mapper_config:
        entities_to_map['source'] = entities_to_map['source'].str.removeprefix(mapper_config['remove_prefix'])
    if 'known_entity_files' in mapper_config:
        triple_reader = get_reader_for_format(kg_config['format'])
        known_entities = set()
        for filename in mapper_config['known_entity_files']:
            for sub, _, _ in triple_reader.read(DATA_DIR / filename):
                known_entities.add(sub)
        entities_to_map.loc[~entities_to_map['source'].isin(known_entities), 'source'] = ''
    mapped_ents = len(entities_to_map[entities_to_map['source'] != ''])
    get_logger().info(f'Found {mapped_ents} entities to map via identity.')
    entities_to_map.to_csv(path_to_entity_mapping, sep='\t', index=False)


if __name__ == "__main__":
    mapper_id = os.environ['KGREAT_STEP']
    with open(KG_DIR / 'config.yaml') as f:
        kg_config = yaml.safe_load(f)
    init_logger(f'mapping-{mapper_id}', kg_config['log_level'])
    mapper_config = kg_config['mapping'][mapper_id]
    perform_identity_mapping(kg_config, mapper_config)
