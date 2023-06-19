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
    dataset_ids = {task_entry['dataset'] for task_entry in task_config.values()}
    # retrieve entities to be mapped from task containers
    _fetch_entity_files(container_manager, temp_dir, dataset_ids)
    # merge entity files of tasks into single entity file
    _merge_entity_files(kg_name, temp_dir, dataset_ids)
    # cleanup
    shutil.rmtree(temp_dir)


def _fetch_entity_files(container_manager: str, temp_dir: Path, dataset_ids: Set[str]):
    for dataset_id in dataset_ids:
        image_name = get_image_name('tasks', dataset_id)
        tmp_container_name = f'tmp_{dataset_id}'
        trigger_container_action(container_manager, 'create', ['--name', tmp_container_name, image_name])
        trigger_container_action(container_manager, 'cp', [f'{tmp_container_name}:/app/entities.tsv', f'{temp_dir}/entities_{dataset_id}.tsv'])
        trigger_container_action(container_manager, 'rm', [tmp_container_name])


def _merge_entity_files(kg_name: str, temp_dir: Path, dataset_ids: Set[str]):
    mapped_ents = []
    mapping_dict = {}
    for dataset_id in dataset_ids:
        ents = pd.read_csv(f'{temp_dir}/entities_{dataset_id}.tsv', header=0, sep='\t')
        _add_entities_to_mapping_dict(ents, mapped_ents, mapping_dict)
    df = pd.DataFrame(mapped_ents)
    df['source'] = ''
    df.to_csv(f'./kg/{kg_name}/entity_mapping.tsv', sep='\t', index=False)


def _add_entities_to_mapping_dict(ents: pd.DataFrame, mapped_ents: list, mapping_dict: dict):
    for _, row in ents.iterrows():
        entity_ids = {k: v for k, v in row.items() if str(k).endswith('_URI') and pd.notnull(v)}
        entity_labels = {k: v for k, v in row.items() if k not in entity_ids and pd.notnull(v)}
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
        mapping_dict_entry.update(entity_ids)
        mapping_dict_entry.update(entity_labels)
        for entity_id in entity_ids.values():
            mapping_dict[entity_id] = mapping_dict_entry
