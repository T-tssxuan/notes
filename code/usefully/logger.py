import logging
import sys

class Logger:
    log_file = ''

    def set_log_file(file_name):
        Tools.log_file = file_name

    def get_logger(name='defualt', level=logging.INFO):
        logger = logging.getLogger(name)
        logger.setLevel(level)

        fmt = '%(asctime)s %(levelname)-8s [' + name + '] %(message)s'
        datefmt = '%Y-%m-%d %H:%M:%S'
        formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

        if Tools.log_file != '':
            handler = logging.FileHandler('log.txt', mode='a')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        screen_handler = logging.StreamHandler(stream=sys.stdout)
        screen_handler.setFormatter(formatter)
        logger.addHandler(screen_handler)
        return logger
