from typing import List
import argparse
from shared.manager.container import perform_image_action, perform_container_action
from shared.manager.prepare_mapping import collect_entities_to_map
from shared.manager.util import Stage, StageAction, load_kg_config, is_local_image


def process_action(kg_name: str, action: StageAction, stage: Stage, steps: List[str], container_manager: str):
    # check configured steps for validity
    stage_config = load_kg_config(kg_name)[stage.value]
    all_steps = list(stage_config)
    invalid_steps = [s for s in steps if s not in all_steps]
    if invalid_steps:
        raise ValueError(f'Found steps that are not defined in the config of the KG: {", ".join(invalid_steps)}. Aborting.')
    steps = steps or all_steps  # use all steps if no explicit ones are provided

    if action == StageAction.PREPARE:
        # currently, only prepare step for "mapping" implemented
        if stage == Stage.MAPPING:
            collect_entities_to_map(kg_name, container_manager)
    elif action in [StageAction.BUILD, StageAction.PUSH, StageAction.PULL]:
        image_names = {stage_config[step_id]['image'] for step_id in steps}
        # trigger build/push actions only for local images
        image_names = {name for name in image_names if action == StageAction.PULL or is_local_image(name)}
        for image_name in image_names:
            perform_image_action(container_manager, action, stage, image_name)
    elif action == StageAction.RUN:
        for step_id in steps:
            perform_container_action(container_manager, action, stage_config[step_id]['image'], kg_name, step_id)
    else:
        raise NotImplementedError(f'Action not implemented: {action.value}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Manage and run images for preprocessing or tasks.')
    parser.add_argument('kg_name', type=str, help='Name of the knowledge graph to use')
    parser.add_argument('action', type=str, help='Action to perform', choices=[a.value for a in StageAction])
    parser.add_argument('--stage', type=str, choices=[s.value for s in Stage], default=None, help='Apply action to this stage (default: all stages)')
    parser.add_argument('--step', type=str, nargs='*', default=[], help='ID of step(s) to perform action on (default: all steps of stage)')
    parser.add_argument('-c', '--container_manager', type=str, help='Name of container manager', choices=['docker', 'podman'], default='docker')
    args = parser.parse_args()
    if args.stage is None and args.step:
        raise ValueError('Invalid input: provided steps without specifying a stage.')

    stages = [Stage(args.stage)] if args.stage is not None else [Stage.MAPPING, Stage.PREPROCESSING, Stage.TASK]
    for stage in stages:
        print(f'Triggering action {args.action} for stage {stage.value}..')
        process_action(args.kg_name, StageAction(args.action), stage, args.step, args.container_manager)
