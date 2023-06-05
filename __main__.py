from typing import List, Optional
import argparse
from util import get_image_name, load_task_config, get_one_task_id_per_task_type, trigger_container_action
from prepare_mapping import collect_entities_to_map


def process_action(kg_name: str, step: str, action: str, task_ids: List[str], container_manager: str):
    if step == 'preprocessing/mapping':
        _perform_mapping_action(kg_name, step, action, container_manager)
    elif step == 'preprocessing/embeddings':
        _perform_embedding_action(kg_name, step, action, container_manager)
    elif step == 'tasks':
        _perform_task_action(kg_name, step, action, container_manager, task_ids)
    else:
        raise NotImplementedError(f'Step not implemented: {step}')


def _perform_mapping_action(kg_name: str, step: str, action: str, container_manager: str):
    if action == 'prepare':
        collect_entities_to_map(kg_name, container_manager)
    else:
        path_to_dockerfile = './shared/preprocessing/mapping/Dockerfile'
        _perform_action(container_manager, action, get_image_name(step), path_to_dockerfile, kg_name)


def _perform_embedding_action(kg_name: str, step: str, action: str, container_manager: str):
    path_to_dockerfile = './shared/preprocessing/embeddings/Dockerfile'
    _perform_action(container_manager, action, get_image_name(step), path_to_dockerfile, kg_name)


def _perform_task_action(kg_name: str, step: str, action: str, container_manager: str, task_ids: List[str]):
    task_config = load_task_config(kg_name)
    if not task_ids:
        # filter tasks such that we have only one per task type (for all actions except 'run', task_id is irrelevant)
        task_ids = list(task_config) if action == 'run' else get_one_task_id_per_task_type(task_config)
    for task_id in task_ids:
        task_type = task_config[task_id]['type']
        image_name = get_image_name(step, task_type)
        path_to_dockerfile = f'./tasks/{task_type}/Dockerfile'
        _perform_action(container_manager, action, image_name, path_to_dockerfile, kg_name, task_id)


def _perform_action(container_manager: str, action: str, image_name: str, path_to_dockerfile: str, kg_name: str, task_id: Optional[str] = None):
    if action == 'build':
        _action_build(container_manager, image_name, path_to_dockerfile)
    elif action == 'push':
        _action_push(container_manager, image_name)
    elif action == 'pull':
        _action_pull(container_manager, image_name)
    elif action == 'prepare':
        pass  # only implemented for mapping, handled separately there.
    elif action == 'run':
        _action_run(container_manager, image_name, kg_name, task_id)


def _action_build(container_manager: str, image_name: str, path_to_dockerfile: str, ):
    trigger_container_action(container_manager, 'build', ['-t', image_name, '-f', path_to_dockerfile, '.'])


def _action_push(container_manager: str, image_name: str):
    trigger_container_action(container_manager, 'push', [image_name])


def _action_pull(container_manager: str, image_name: str):
    trigger_container_action(container_manager, 'pull', [image_name])


def _action_run(container_manager: str, image_name: str, kg_name: Optional[str], task_id: Optional[str]):
    if kg_name is None:
        raise ValueError('Parameter "kg_name" must be specified for running a container!')
    params = ['--mount', f'type=bind,src=./kg/{kg_name},target=/app/kg']
    if task_id is not None:
        params += ['-e', f'KGREAT_TASK={task_id}']
    params += [image_name]
    trigger_container_action(container_manager, 'run', params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Manage and run images for preprocessing or tasks.')
    parser.add_argument('kg_name', type=str, help='Name of the knowledge graph to use')
    parser.add_argument('step', type=str, help='Apply action to this step', choices=['preprocessing/mapping', 'preprocessing/embeddings', 'tasks'])
    parser.add_argument('action', type=str, help='Action performed on image', choices=['build', 'push', 'pull', 'prepare', 'run'])
    parser.add_argument('-t', '--task_ids', type=str, nargs='*', help='IDs of tasks to perform action on (default is all)')
    parser.add_argument('-c', '--container_manager', type=str, help='Name of container manager', choices=['docker', 'podman'], default='docker')
    args = parser.parse_args()
    process_action(args.kg_name, args.step, args.action, args.task_ids, args.container_manager)
