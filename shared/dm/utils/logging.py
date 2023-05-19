import logging
from utils.io import get_kg_result_path


def get_logger():
    return logging.getLogger('dm')


def init_logger(run_id: str, task_id: str, log_level: str):
    log_filepath = get_kg_result_path(run_id) / f'{task_id}.log'
    log_handler = logging.FileHandler(log_filepath, 'a', 'utf-8')
    log_handler.setFormatter(logging.Formatter('%(asctime)s|%(levelname)s|%(module)s->%(funcName)s: %(message)s'))
    log_handler.setLevel(log_level)
    logger = get_logger()
    logger.addHandler(log_handler)
    logger.setLevel(log_level)
