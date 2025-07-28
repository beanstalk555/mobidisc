import logging


def setup_logger(
    name,
    log_file,
    formatter="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    filemode="w",
):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file, mode=filemode)
    handler.setFormatter(logging.Formatter(formatter))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger
