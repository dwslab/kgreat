import os
import yaml
import pandas as pd
import logging
import datetime
from pathlib import Path
from importer import get_reader_for_format


KG_DIR = Path('./kg')
DATA_DIR = KG_DIR / 'data'


def perform_sameas_mapping(kg_config: dict, mapper_config: dict):
    # parse entities to map
    path_to_entity_mapping = KG_DIR / 'entity_mapping.tsv'
    if not path_to_entity_mapping.is_file():
        raise FileNotFoundError('Could not find entity mapping file. Did you forget to `prepare` the mapping stage?')
    entities_to_map = pd.read_csv(path_to_entity_mapping, sep='\t', header=0, dtype=str)
    # parse sameas links and apply to entities
    _get_logger().info('Parsing sameAs links from files..')
    sameas_mapping = _parse_sameas_links_from_files(kg_config, mapper_config)
    _get_logger().info(f'Found {len(sameas_mapping)} sameAs links.')
    uri_columns = [col for col in entities_to_map.columns if col.endswith('_URI')]
    mapped_ents = 0
    for idx, row in entities_to_map.iterrows():
        if not pd.isnull(row['source']):
            continue
        for col in uri_columns:
            if row[col] in sameas_mapping:
                entities_to_map.loc[idx, 'source'] = sameas_mapping[row[col]]
                mapped_ents += 1
                continue
    _get_logger().info(f'Found {mapped_ents} entities to map via sameAs links.')
    # write updated entity mapping
    if mapped_ents > 0:
        _get_logger().info('Writing updated entity mapping file..')
        entities_to_map.to_csv(path_to_entity_mapping, sep='\t', index=False)


def _parse_sameas_links_from_files(kg_config: dict, mapper_config: dict) -> dict:
    # initialize valid predicates and kg prefix
    sameas_predicates = {'http://www.w3.org/2002/07/owl#sameAs'}
    if 'additional_predicates' in mapper_config:
        sameas_predicates.update(set(mapper_config['additional_predicates']))
    kg_prefix = mapper_config['kg_prefix']
    # parse links from files
    triple_reader = get_reader_for_format(kg_config['format'])
    sameas_mapping = {}
    for filename in mapper_config['input_files']:
        for sub, pred, obj in triple_reader.read(DATA_DIR / filename):
            if pred not in sameas_predicates:
                continue
            if sub.startswith(kg_prefix):
                sameas_mapping[obj] = sub
            elif obj.startswith(kg_prefix):
                sameas_mapping[sub] = obj
    return sameas_mapping


def _get_logger():
    return logging.getLogger('mapping/sameAs')


def _init_logger(log_level: str):
    log_filepath = KG_DIR / '{}_mapping-sameAs.log'.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    log_handler = logging.FileHandler(log_filepath, 'a', 'utf-8')
    log_handler.setFormatter(logging.Formatter('%(asctime)s|%(levelname)s|%(module)s->%(funcName)s: %(message)s'))
    log_handler.setLevel(log_level)
    logger = _get_logger()
    logger.addHandler(log_handler)
    logger.setLevel(log_level)


if __name__ == "__main__":
    mapper_id = os.environ['KGREAT_STEP']
    with open(KG_DIR / 'config.yaml') as f:
        kg_config = yaml.safe_load(f)
    _init_logger(kg_config['log_level'])
    mapper_config = kg_config['mapping'][mapper_id]
    perform_sameas_mapping(kg_config, mapper_config)
