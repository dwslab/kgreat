import yaml
from enum import Enum


IMAGE_PREFIX = 'gitlab.dws.informatik.uni-mannheim.de:5050/nheist/kgreat'


class Stage(Enum):
    MAPPING = 'mapping'
    PREPROCESSING = 'preprocessing'
    TASK = 'task'


class StageAction(Enum):
    PREPARE = 'prepare'
    BUILD = 'build'
    PUSH = 'push'
    PULL = 'pull'
    RUN = 'run'


def is_local_image(image: str) -> bool:
    return image.startswith(IMAGE_PREFIX)


def get_image_step(image: str) -> str:
    return image[image.rfind('/') + 1:]


def get_path_to_dockerfile(stage: Stage, image_name: str):
    image_step = get_image_step(image_name)
    if stage in [Stage.MAPPING, Stage.PREPROCESSING]:
        return f'shared/{stage.value}/{image_step}/Dockerfile'
    elif stage == Stage.TASK:
        return f'dataset/{image_step}/Dockerfile'
    else:
        raise NotImplementedError(f'Dockerfile retrieval for stage not implemented: {stage.value}')


def load_kg_config(kg_name: str) -> dict:
    with open(f'./kg/{kg_name}/config.yaml') as f:
        return yaml.safe_load(f)
