from typing import List
import datetime
import subprocess
from shared.manager.util import get_path_to_dockerfile, Stage, StageAction


def perform_image_action(container_manager: str, action: StageAction, stage: Stage, image_name: str):
    if action == StageAction.BUILD:
        params = ['-t', image_name, '-f', get_path_to_dockerfile(stage, image_name), '.']
    elif action in [StageAction.PUSH, StageAction.PULL]:
        params = [image_name]
    else:
        raise NotImplementedError(f'Image action not implemented: {action.value}')
    _trigger_container_action(container_manager, action.value, params)


def perform_container_action(container_manager: str, action: StageAction, image_name: str, kg_name: str, step_id: str):
    if action != StageAction.RUN:
        raise NotImplementedError(f'Container action not implemented: {action.value}')
    params = ['--mount', f'type=bind,src=./kg/{kg_name},target=/app/kg', '-e', f'KGREAT_STEP={step_id}', image_name]
    _trigger_container_action(container_manager, action.value, params)


def _trigger_container_action(container_manager: str, action: str, params: List[str]) -> str:
    command = [container_manager, action] + params
    print(f'{datetime.datetime.now()} | Running command -> {" ".join(command)}')
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = process.communicate()[1].decode()
    print(output)
    return output
