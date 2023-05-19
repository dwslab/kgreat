import datetime
from typing import List, Optional
import argparse
import subprocess
from pathlib import Path
import yaml


IMAGE_PREFIX = 'gitlab.dws.informatik.uni-mannheim.de:5050/nheist/kgreat/'


def process_action(action: str, step: str, kg_name: Optional[str], task_ids: List[str], container_manager: str):
    if step == 'preprocessing.embeddings':
        _perform_embedding_action(action, kg_name, container_manager)
    elif step == 'tasks':
        _perform_task_action(action, kg_name, task_ids, container_manager)
    else:
        raise NotImplementedError(f'Step not implemented: {step}')


def _perform_embedding_action(action: str, kg_name: Optional[str], container_manager: str):
    image_name = IMAGE_PREFIX + 'preprocessing/embeddings'
    path_to_dockerfile = './shared/preprocessing/embeddings/Dockerfile'
    _perform_action(container_manager, action, image_name, path_to_dockerfile, kg_name)


def _perform_task_action(action: str, kg_name: Optional[str], task_ids: List[str], container_manager: str):
    task_config = _load_task_config(kg_name)
    task_ids = task_ids or list(task_config)
    if action != 'run':
        # filter tasks such that we have only one per task type (for all actions except 'run', task_id is irrelevant)
        filtered_task_ids = []
        known_task_types = set()
        for task_id in task_ids:
            task_type = task_config[task_id]['type']
            if task_type not in known_task_types:
                filtered_task_ids.append(task_id)
                known_task_types.add(task_type)
        task_ids = filtered_task_ids
    for task_id in task_ids:
        task_type = task_config[task_id]['type']
        image_name = f'{IMAGE_PREFIX}tasks/{task_type.lower()}'
        path_to_dockerfile = f'./tasks/{task_type}/Dockerfile'
        _perform_action(container_manager, action, image_name, path_to_dockerfile, kg_name, task_id)


def _load_task_config(kg_name: Optional[str]) -> dict:
    # use default config if no kg specified
    filepath = Path('./example_config.yaml' if kg_name is None else f'./kg/{kg_name}/config.yaml')
    with open(filepath) as f:
        return yaml.safe_load(f)['tasks']


def _perform_action(container_manager: str, action: str, image_name: str, path_to_dockerfile: str, kg_name: str, task_id: Optional[str] = None):
        if action == 'build':
            _action_build(container_manager, image_name, path_to_dockerfile)
        elif action == 'push':
            _action_push(container_manager, image_name)
        elif action == 'pull':
            _action_pull(container_manager, image_name)
        elif action == 'run':
            _action_run(container_manager, image_name, kg_name, task_id)


def _action_build(container_manager: str, image_name: str, path_to_dockerfile: str, ):
    _action(container_manager, 'build', ['-t', image_name, '-f', path_to_dockerfile, '.'])


def _action_push(container_manager: str, image_name: str):
    _action(container_manager, 'push', [image_name])


def _action_pull(container_manager: str, image_name: str):
    _action(container_manager, 'pull', [image_name])


def _action_run(container_manager: str, image_name: str, kg_name: Optional[str], task_id: Optional[str]):
    if kg_name is None:
        raise ValueError('Parameter "kg_name" must be given for running a container!')
    params = ['--mount', f'type=bind,src=./kg/{kg_name},target=/app/kg']
    if task_id is not None:
        params += ['-e', f'KGREAT_TASK={task_id}']
    params += [image_name]
    _action(container_manager, 'run', params)


def _action(container_manager: str, action: str, params: List[str]):
    command = [container_manager, action] + params
    print(f'{datetime.datetime.now()} | Running command -> {" ".join(command)}')
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(process.communicate()[1].decode())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Manage and run images for preprocessing or tasks.')
    parser.add_argument('action', type=str, help='Action performed on image', choices=['build', 'push', 'pull', 'run'])
    parser.add_argument('step', type=str, help='Use only images of this step', choices=['preprocessing.embeddings', 'tasks'])
    parser.add_argument('-k', '--kg_name', type=str, help='Name of the knowledge graph to use')
    parser.add_argument('-t', '--task_ids', type=str, nargs='*', help='IDs of tasks to perform action on (default is all)')
    parser.add_argument('-c', '--container_manager', type=str, help='Name of container manager', choices=['docker', 'podman'], default='docker')
    args = parser.parse_args()
    process_action(args.action, args.step, args.kg_name, args.task_ids, args.container_manager)
