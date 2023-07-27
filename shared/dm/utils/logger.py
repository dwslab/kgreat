import datetime
from pathlib import Path
import logging


LOGGER_NAME = None


def init_logger(log_level: str, run_id: str, task_id: str):
    global LOGGER_NAME
    LOGGER_NAME = f'task-{task_id}'
    log_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_folder = Path(f'kg/log/{run_id}')
    log_folder.mkdir(parents=True, exist_ok=True)  # make sure log folder exists
    log_file = log_folder / f'kg/log/{run_id}/{log_time}_{LOGGER_NAME}.log'
    log_handler = logging.FileHandler(log_file, 'a', 'utf-8')
    log_handler.setFormatter(logging.Formatter('%(asctime)s|%(levelname)s|%(module)s->%(funcName)s: %(message)s'))
    log_handler.setLevel(log_level)
    logger = get_logger()
    logger.addHandler(log_handler)
    logger.setLevel(log_level)


def get_logger():
    return logging.getLogger(LOGGER_NAME)
