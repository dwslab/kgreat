import logging
import platform
import datetime


LOGGER_NAME = None


class HostnameFilter(logging.Filter):
    hostname = platform.node()

    def filter(self, record):
        record.hostname = HostnameFilter.hostname
        return True


def init_logger(logger_name: str, log_level: str):
    global LOGGER_NAME
    LOGGER_NAME = logger_name
    log_filepath = 'kg/{}_{}.log'.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'), logger_name)
    log_handler = logging.FileHandler(log_filepath, 'a', 'utf-8')
    log_handler.setFormatter(logging.Formatter('%(asctime)s|%(hostname)s|%(levelname)s|%(module)s->%(funcName)s: %(message)s'))
    log_handler.setLevel(log_level)
    logger = get_logger()
    logger.addHandler(log_handler)
    logger.setLevel(log_level)


def get_logger():
    return logging.getLogger(LOGGER_NAME)
