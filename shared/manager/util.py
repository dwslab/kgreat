from typing import Optional, List
import yaml
import datetime
import subprocess


IMAGE_PREFIX = 'gitlab.dws.informatik.uni-mannheim.de:5050/nheist/kgreat'


def get_image_name(step_id: str, suffix: Optional[str] = None) -> str:
    image_name = f'{IMAGE_PREFIX}/{step_id}'
    return image_name if suffix is None else image_name + f'/{suffix.lower()}'


def load_kg_config(kg_name: str) -> dict:
    with open(f'./kg/{kg_name}/config.yaml') as f:
        return yaml.safe_load(f)


def get_one_step_per_type(config: dict) -> List[str]:
    step_ids = []
    known_step_types = set()
    for step_id, step_entry in config.items():
        step_type = step_entry['type']
        if step_type not in known_step_types:
            step_ids.append(step_id)
            known_step_types.add(step_type)
    return step_ids


def trigger_container_action(container_manager: str, action: str, params: List[str]) -> str:
    command = [container_manager, action] + params
    print(f'{datetime.datetime.now()} | Running command -> {" ".join(command)}')
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = process.communicate()[1].decode()
    print(output)
    return output
