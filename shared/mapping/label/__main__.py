from typing import Dict, Set
from collections import defaultdict
import os
import yaml
import math
from logger import init_logger, get_logger
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz import process
from pathlib import Path
from importer import get_reader_for_format


KG_DIR = Path('./kg')
DATA_DIR = KG_DIR / 'data'


def perform_label_mapping(kg_config: dict, mapper_config: dict):
    # parse entities to map
    path_to_entity_mapping = KG_DIR / 'entity_mapping.tsv'
    if not path_to_entity_mapping.is_file():
        raise FileNotFoundError('Could not find entity mapping file. Did you forget to `prepare` the mapping stage?')
    entities_to_map = pd.read_csv(path_to_entity_mapping, sep='\t', header=0, dtype=str)
    # ignore entities that are either mapped already or have no valid label
    entities_with_label = entities_to_map[(entities_to_map['source'].isnull()) & (entities_to_map['label'].notnull())]
    get_logger().info(f'Found {len(entities_with_label)} potential entities to map.')
    get_logger().info('Parsing KG entities and labels from files..')
    kg_entity_labels = _parse_labels_from_files(kg_config, mapper_config)
    get_logger().info(f'Found {len(kg_entity_labels)} labels to map against.')
    mapped_entities = {}
    # find entities with exact match
    get_logger().info('Looking for exact matches..')
    definitive_matches_only = 'definitive_matches_only' in mapper_config and mapper_config['definitive_matches_only']
    _find_exact_matches(entities_with_label, kg_entity_labels, definitive_matches_only, mapped_entities)
    num_exact_matches = len(mapped_entities)
    get_logger().info(f'Found {num_exact_matches} exact matches.')
    # find entities with fuzzy matching
    similarity_threshold = mapper_config['similarity_threshold']
    if not definitive_matches_only and similarity_threshold < 1:
        get_logger().info(f'Looking for fuzzy matches with similarity > {similarity_threshold}..')
        entities_with_label = entities_with_label[~entities_with_label.index.isin(set(mapped_entities))]
        # process entities in chunks to limit memory consumption
        chunk_size = mapper_config['chunk_size'] if 'chunk_size' in mapper_config else 10000
        num_chunks = math.ceil(len(entities_with_label) / chunk_size)
        for chunk in range(num_chunks):
            get_logger().debug(f'Processing chunk {chunk+1} of {num_chunks}..')
            chunk_start = chunk * chunk_size
            _find_fuzzy_matches(entities_with_label.iloc[chunk_start:chunk_start+chunk_size, :], kg_entity_labels, similarity_threshold, mapped_entities)
        get_logger().info(f'Found {len(mapped_entities) - num_exact_matches} fuzzy matches.')
    # write updated entity mapping
    get_logger().info('Writing updated entity mapping file..')
    entities_to_map.loc[list(mapped_entities), 'source'] = list(mapped_entities.values())
    entities_to_map.to_csv(path_to_entity_mapping, sep='\t', index=False)


def _parse_labels_from_files(kg_config: dict, mapper_config: dict) -> Dict[str, Set[str]]:
    # initialize valid predicates and kg prefix
    label_predicates = set(mapper_config['label_predicates'])
    # parse labels from files
    triple_reader = get_reader_for_format(kg_config['format'])
    kg_entity_labels = defaultdict(set)
    for filename in mapper_config['input_files']:
        for sub, pred, obj in triple_reader.read(DATA_DIR / filename):
            if pred in label_predicates:
                kg_entity_labels[obj].add(sub)
    return kg_entity_labels


def _find_exact_matches(entities_with_label: pd.DataFrame, kg_entity_labels: dict, definitive_matches_only: bool, mapped_entities: dict):
    for ent_idx, ent_label in entities_with_label['label'].items():
        if ent_label in kg_entity_labels:
            possible_ents = list(kg_entity_labels[ent_label])
            if definitive_matches_only and len(possible_ents) > 1:
                continue
            mapped_entities[ent_idx] = possible_ents[0]


def _find_fuzzy_matches(entities_with_label: pd.DataFrame, kg_entity_labels: dict, similarity_threshold: float, mapped_entities: dict):
    kg_labels, kg_entities = list(kg_entity_labels), list(kg_entity_labels.values())
    get_logger().debug(f'Computing token sort ratio: {len(entities_with_label)}x{len(kg_labels)}..')
    mapping_result = process.cdist(entities_with_label['label'].values, kg_labels, scorer=fuzz.token_sort_ratio, score_cutoff=similarity_threshold, workers=kg_config['max_cpus'])
    get_logger().debug(f'Finding best-matching entities..')
    for ent_idx, scores, matched_entity_idx in zip(entities_with_label.index, mapping_result, mapping_result.argmax(axis=-1)):
        if scores[matched_entity_idx] < similarity_threshold:
            continue
        mapped_entities[ent_idx] = list(kg_entities[matched_entity_idx])[0]


if __name__ == "__main__":
    mapper_id = os.environ['KGREAT_STEP']
    with open(KG_DIR / 'config.yaml') as f:
        kg_config = yaml.safe_load(f)
    init_logger(f'mapping-{mapper_id}', kg_config['log_level'], kg_config['run_id'])
    mapper_config = kg_config['mapping'][mapper_id]
    perform_label_mapping(kg_config, mapper_config)
