from typing import List
import argparse
from shared.manager.container import perform_action
from shared.manager.prepare_mapping import collect_entities_to_map
from shared.manager.util import get_image_name, load_kg_config, get_one_step_per_attr


def process_action(kg_name: str, action: str, stage: str, steps: List[str], container_manager: str):
    if stage == 'mapping':
        _perform_mapping_action(kg_name, action, stage, steps, container_manager)
    elif stage == 'preprocessing/embedding':
        _perform_embedding_action(kg_name, action, stage, steps, container_manager)
    elif stage == 'preprocessing/embedding-speedup':
        _perform_embedding_speedup_action(kg_name, action, stage, steps, container_manager)
    elif stage == 'tasks':
        _perform_task_action(kg_name, action, stage, steps, container_manager)
    else:
        raise NotImplementedError(f'Stage not implemented: {stage}')


def _perform_mapping_action(kg_name: str, action: str, stage: str, steps: List[str], container_manager: str):
    if action == 'prepare':
        collect_entities_to_map(kg_name, container_manager)
        return
    mapping_config = load_kg_config(kg_name)['mapping']
    if not steps:
        # handling one of every step type (= image) is enough if we do not run them
        steps = list(mapping_config) if action == 'run' else get_one_step_per_attr(mapping_config, 'type')
    for mapper_id in steps:
        mapper_type = mapping_config[mapper_id]['type']
        image_name = get_image_name(stage, mapper_type)
        path_to_dockerfile = f'shared/mapping/{mapper_type}/Dockerfile'
        perform_action(container_manager, action, image_name, path_to_dockerfile, kg_name, mapper_id)


def _perform_embedding_action(kg_name: str, action: str, stage: str, steps: List[str], container_manager: str):
    path_to_dockerfile = 'shared/preprocessing/embedding/Dockerfile'
    perform_action(container_manager, action, get_image_name(stage), path_to_dockerfile, kg_name)


def _perform_embedding_speedup_action(kg_name: str, action: str, stage: str, steps: List[str], container_manager: str):
    path_to_dockerfile = 'shared/preprocessing/embedding-speedup/Dockerfile'
    perform_action(container_manager, action, get_image_name(stage), path_to_dockerfile, kg_name)


def _perform_task_action(kg_name: str, action: str, stage: str, steps: List[str], container_manager: str):
    task_config = load_kg_config(kg_name)['tasks']
    if not steps:
        # handling one of every step type (= image) is enough if we do not run them
        steps = list(task_config) if action == 'run' else get_one_step_per_attr(task_config, 'dataset')
    for task_id in steps:
        dataset_id = task_config[task_id]['dataset']
        image_name = get_image_name(stage, dataset_id)
        path_to_dockerfile = f'datasets/{dataset_id}/Dockerfile'
        perform_action(container_manager, action, image_name, path_to_dockerfile, kg_name, task_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Manage and run images for preprocessing or tasks.')
    parser.add_argument('kg_name', type=str, help='Name of the knowledge graph to use')
    parser.add_argument('action', type=str, help='Action to perform', choices=['build', 'push', 'pull', 'prepare', 'run'])
    parser.add_argument('stage', type=str, help='Apply action to this stage', choices=['mapping', 'preprocessing/embedding', 'preprocessing/embedding-speedup', 'tasks'])
    parser.add_argument('-s', '--step', type=str, nargs='*', help='ID of step(s) to perform action on (default is: all steps of stage)')
    parser.add_argument('-c', '--container_manager', type=str, help='Name of container manager', choices=['docker', 'podman'], default='docker')
    args = parser.parse_args()
    process_action(args.kg_name, args.action, args.stage, args.step, args.container_manager)
