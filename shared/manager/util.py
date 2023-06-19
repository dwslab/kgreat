from typing import Optional, List
import yaml


IMAGE_PREFIX = 'ANONYMIZED'


def get_image_name(step_id: str, suffix: Optional[str] = None) -> str:
    image_name = f'{IMAGE_PREFIX}/{step_id}'
    return image_name if suffix is None else image_name + f'/{suffix.lower()}'


def load_kg_config(kg_name: str) -> dict:
    with open(f'./kg/{kg_name}/config.yaml') as f:
        return yaml.safe_load(f)


def get_one_step_per_attr(config: dict, attr_name: str) -> List[str]:
    step_ids = []
    known_step_attrs = set()
    for step_id, step_entry in config.items():
        step_attr = step_entry[attr_name]
        if step_attr not in known_step_attrs:
            step_ids.append(step_id)
            known_step_attrs.add(step_attr)
    return step_ids
