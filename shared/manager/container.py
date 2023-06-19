from typing import Optional
import datetime
import subprocess


def perform_action(container_manager: str, action: str, image_name: str, path_to_dockerfile: str, kg_name: str, step_id: Optional[str] = None):
    if action == 'build':
        _action_build(container_manager, image_name, path_to_dockerfile)
    elif action == 'push':
        _action_push(container_manager, image_name)
    elif action == 'pull':
        _action_pull(container_manager, image_name)
    elif action == 'prepare':
        pass  # only implemented for mapping, handled separately there.
    elif action == 'run':
        _action_run(container_manager, image_name, kg_name, step_id)


def _action_build(container_manager: str, image_name: str, path_to_dockerfile: str, ):
    _trigger_container_action(container_manager, 'build', ['-t', image_name, '-f', path_to_dockerfile, '.'])


def _action_push(container_manager: str, image_name: str):
    _trigger_container_action(container_manager, 'push', [image_name])


def _action_pull(container_manager: str, image_name: str):
    _trigger_container_action(container_manager, 'pull', [image_name])


def _action_run(container_manager: str, image_name: str, kg_name: str, step_id: Optional[str]):
    params = ['--mount', f'type=bind,src=./kg/{kg_name},target=/app/kg']
    if step_id is not None:
        params += ['-e', f'KGREAT_STEP={step_id}']
    params += [image_name]
    _trigger_container_action(container_manager, 'run', params)


def _trigger_container_action(container_manager: str, action: str, params: List[str]) -> str:
    command = [container_manager, action] + params
    print(f'{datetime.datetime.now()} | Running command -> {" ".join(command)}')
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = process.communicate()[1].decode()
    print(output)
    return output
