from typing import Set
import tempfile
import pandas as pd
import shutil
from pathlib import Path
from shared.manager.container import _trigger_container_action
from shared.manager.util import load_kg_config, get_image_step


def collect_entities_to_map(kg_name: str, container_manager: str):
    temp_dir = Path(tempfile.mkdtemp())
    # find relevant tasks
    task_config = load_kg_config(kg_name)['task']
    image_names = {task_entry['image'] for task_entry in task_config.values()}
    # retrieve entities to be mapped from task containers
    _fetch_entity_files(container_manager, temp_dir, image_names)
    # merge entity files of tasks into single entity file
    _merge_entity_files(kg_name, temp_dir, image_names)
    # cleanup
    shutil.rmtree(temp_dir)


def _fetch_entity_files(container_manager: str, temp_dir: Path, image_names: Set[str]):
    for image_name in image_names:
        image_step = get_image_step(image_name)
        tmp_container_name = f'{temp_dir.stem}_{image_step}'
        _trigger_container_action(container_manager, 'create', ['--name', tmp_container_name, image_name])
        _trigger_container_action(container_manager, 'cp', [f'{tmp_container_name}:/app/entities.tsv', f'{temp_dir}/entities_{image_step}.tsv'])
        _trigger_container_action(container_manager, 'rm', [tmp_container_name])


def _merge_entity_files(kg_name: str, temp_dir: Path, image_names: Set[str]):
    mapped_ents = []
    mapping_dict = {}
    for image_name in image_names:
        ents = pd.read_csv(f'{temp_dir}/entities_{get_image_step(image_name)}.tsv', header=0, sep='\t')
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
