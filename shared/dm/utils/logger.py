import logging
import platform
from .io import get_kg_result_path


LOGGER_NAME = None


class HostnameFilter(logging.Filter):
    hostname = platform.node()

    def filter(self, record):
        record.hostname = HostnameFilter.hostname
        return True


def get_logger():
    return logging.getLogger(LOGGER_NAME)


def init_logger(run_id: str, task_id: str, log_level: str):
    global LOGGER_NAME
    LOGGER_NAME = f'task-{task_id}'
    log_filepath = get_kg_result_path(run_id) / f'{task_id}.log'
    log_handler = logging.FileHandler(log_filepath, 'a', 'utf-8')
    log_handler.setFormatter(logging.Formatter('%(asctime)s|%(hostname)s|%(levelname)s|%(module)s->%(funcName)s: %(message)s'))
    log_handler.setLevel(log_level)
    logger = get_logger()
    logger.addHandler(log_handler)
    logger.setLevel(log_level)
