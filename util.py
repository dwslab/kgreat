from typing import Optional, List
import yaml
import datetime
import subprocess


IMAGE_PREFIX = 'gitlab.dws.informatik.uni-mannheim.de:5050/nheist/kgreat'


def get_image_name(step_id: str, suffix: Optional[str] = None) -> str:
    image_name = f'{IMAGE_PREFIX}/{step_id}'
    return image_name if suffix is None else image_name + f'/{suffix.lower()}'


def load_task_config(kg_name: str) -> dict:
    with open(f'./kg/{kg_name}/config.yaml') as f:
        return yaml.safe_load(f)['tasks']


def get_one_task_id_per_task_type(task_config: dict) -> List[str]:
    task_ids = []
    known_task_types = set()
    for task_id, task_entry in task_config.items():
        task_type = task_entry['type']
        if task_type not in known_task_types:
            task_ids.append(task_id)
            known_task_types.add(task_type)
    return task_ids


def trigger_container_action(container_manager: str, action: str, params: List[str]) -> str:
    command = [container_manager, action] + params
    print(f'{datetime.datetime.now()} | Running command -> {" ".join(command)}')
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = process.communicate()[1].decode()
    print(output)
    return output
