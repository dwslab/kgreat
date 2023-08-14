from typing import List
from logger import get_logger
from importer import get_reader_for_format
from pathlib import Path
import csv


KG_DIR = Path('./kg')
DATA_DIR = KG_DIR / 'data'


NT_PREFIX_ENTITY = 'e:'
NT_PREFIX_RELATION = 'r:'


def convert_graph_data(kg_format: str, input_files: List[str], target_folder: Path, output_format: str = 'tsv') -> int:
    get_logger().info(f'Converting input of format {kg_format} to dict-{output_format} format.')
    if output_format not in ['tsv', 'nt']:
        raise NotImplementedError(f'Output format not implemented: {output_format}')
    reader = get_reader_for_format(kg_format)
    # gather entities, relations, triples
    entities, relations, triples = {}, {}, []
    for filename in input_files:
        path = DATA_DIR / filename
        if not path.is_file():
            get_logger().info(f'Could not parse file "{filename}" as it is not contained in the data folder.')
            continue
        for s, r, o in reader.read(path):
            s_idx = _get_or_create_idx(s, entities)
            r_idx = _get_or_create_idx(r, relations)
            o_idx = _get_or_create_idx(o, entities)
            triples.append((s_idx, r_idx, o_idx))
    # persist in new format
    _write_entity_dict_file(entities, target_folder, output_format)
    _write_relation_dict_file(relations, target_folder, output_format)
    _write_train_file(triples, target_folder, output_format)
    return len(triples)


def _get_or_create_idx(value: str, value_dict: dict) -> str:
    if value not in value_dict:
        value_dict[value] = len(value_dict)
    return value_dict[value]


def _write_entity_dict_file(data: dict, target_folder: Path, output_format: str):
    prefix = NT_PREFIX_ENTITY if output_format == 'nt' else ''
    _write_dict_file(data, target_folder / 'entities.dict', prefix)


def _write_relation_dict_file(data: dict, target_folder: Path, output_format: str):
    prefix = NT_PREFIX_RELATION if output_format == 'nt' else ''
    _write_dict_file(data, target_folder / 'relations.dict', prefix)


def _write_dict_file(data: dict, filepath: Path, prefix: str):
    with open(filepath, mode='w', newline='') as f:
        csvwriter = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        for item, idx in data.items():
            csvwriter.writerow([item, prefix + str(idx)])


def _write_train_file(triples: list, target_folder: Path, output_format: str):
    with open(target_folder / f'train.{output_format}', mode='w', newline='') as f:
        csvwriter = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        for s, p, o in triples:
            if output_format == 'tsv':
                csvwriter.writerow([s, p, o])
            elif output_format == 'nt':
                f.write(f'<{NT_PREFIX_ENTITY}{s}> <{NT_PREFIX_RELATION}{p}> <{NT_PREFIX_ENTITY}{o}> .\n')
