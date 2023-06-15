from typing import Dict, Set
from collections import defaultdict
import os
import yaml
import numpy as np
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
    kg_entity_labels = _parse_labels_from_files(kg_config, mapper_config)
    mapped_entities = {}
    # find entities with exact match
    definitive_matches_only = 'definitive_matches_only' in mapper_config and mapper_config['definitive_matches_only']
    for ent_idx, ent_label in entities_with_label['label'].items():
        if ent_label in kg_entity_labels:
            possible_ents = list(kg_entity_labels[ent_label])
            if definitive_matches_only and len(possible_ents) > 1:
                continue
            mapped_entities[ent_idx] = possible_ents[0]
    # find entities with fuzzy matching
    similarity_threshold = mapper_config['similarity_threshold']
    if not definitive_matches_only and similarity_threshold < 1:
        entities_with_label = entities_with_label[~entities_with_label.index.isin(set(mapped_entities))]
        kg_labels, kg_entities = list(kg_entity_labels), list(kg_entity_labels.values())
        mapping_result = process.cdist(entities_with_label['label'].values, kg_labels, scorer=fuzz.token_sort_ratio, score_cutoff=similarity_threshold, workers=kg_config['max_cpus'])
        for ent_idx, scores in zip(entities_with_label.index, mapping_result):
            if max(scores) < similarity_threshold:
                continue
            matched_entity_idx = np.argsort(-scores)[0]
            mapped_entities[ent_idx] = list(kg_entities[matched_entity_idx])[0]
    # write updated entity mapping
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


if __name__ == "__main__":
    mapper_id = os.environ['KGREAT_STEP']
    with open(KG_DIR / 'config.yaml') as f:
        kg_config = yaml.safe_load(f)
    mapper_config = kg_config['mapping'][mapper_id]
    perform_label_mapping(kg_config, mapper_config)
