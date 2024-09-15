import logging

def get_logger(filename, verbosity=1):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename)
    fh.setFormatter(formatter)


    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger
