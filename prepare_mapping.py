from typing import Set
import tempfile
import pandas as pd
import shutil
from pathlib import Path
from util import load_kg_config, get_image_name, trigger_container_action


def collect_entities_to_map(kg_name: str, container_manager: str):
    temp_dir = Path(tempfile.mkdtemp())
    # find relevant tasks
    task_config = load_kg_config(kg_name)['tasks']
    task_types = {task_entry['type'] for task_entry in task_config.values()}
    # retrieve entities to be mapped from task containers
    _fetch_entity_files(container_manager, temp_dir, task_types)
    # merge entity files of tasks into single entity file
    _merge_entity_files(kg_name, temp_dir, task_types)
    # cleanup
    shutil.rmtree(temp_dir)


def _fetch_entity_files(container_manager: str, temp_dir: Path, task_types: Set[str]):
    for task_type in task_types:
        image_name = get_image_name('tasks', task_type)
        tmp_container_name = f'tmp_{task_type}'
        trigger_container_action(container_manager, 'create', ['--name', tmp_container_name, image_name])
        trigger_container_action(container_manager, 'cp', [f'{tmp_container_name}:/app/entities.tsv', f'{temp_dir}/entities_{task_type}.tsv'])
        trigger_container_action(container_manager, 'rm', [tmp_container_name])


def _merge_entity_files(kg_name: str, temp_dir: Path, task_types: Set[str]):
    mapped_ents = []
    mapping_dict = {}
    for task_type in task_types:
        ents = pd.read_csv(f'{temp_dir}/entities_{task_type}.tsv', header=0, sep='\t')
        _add_entities_to_mapping_dict(ents, mapped_ents, mapping_dict)
    df = pd.DataFrame(mapped_ents)
    df['source'] = ''
    df.to_csv(f'./kg/{kg_name}/entity_mapping.tsv', sep='\t', index=False)


def _add_entities_to_mapping_dict(ents: pd.DataFrame, mapped_ents: list, mapping_dict: dict):
    for _, row in ents.iterrows():
        entity_ids = {k: v for k, v in row.items() if str(k).endswith('_URI') and v}
        entity_labels = {k: v for k, v in row.items() if k not in entity_ids and v}
        # use existing mapping_dict entry (if existing), otherwise create new and add to mapped ents
        mapping_dict_entry = {}
        found_existing = False
        for entity_id in entity_ids.values():
            if entity_id in mapping_dict:
                mapping_dict_entry = mapping_dict[entity_id]
                found_existing = True
                break
        if not found_existing:
            mapped_ents.append(mapping_dict_entry)
        # update and index mapping-dict entry
        mapping_dict_entry |= entity_ids
        mapping_dict_entry |= entity_labels
        for entity_id in entity_ids.values():
            mapping_dict[entity_id] = mapping_dict_entry
