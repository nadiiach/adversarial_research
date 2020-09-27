import logging
from os.path import join
import datetime as dt
from os import listdir, remove

logger = None

def get_logger(level=None):
    global logger
    if logger is None:
        logger = init_logger(logging.DEBUG if level is None else level, "logs")
    return logger

def init_logger(logging_level, log_dir, keep_max_logfiles=20):
    import os
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    del_logs = sorted([int(x[:-4]) for x in listdir(log_dir) if ".log" in x], reverse=True)[keep_max_logfiles:]

    for l in del_logs:
        try:
            remove(join(log_dir, str(l) + ".log"))
        except Exception as e:
            continue

    b = dt.datetime.now()
    a = dt.datetime(2020, 4, 1, 0, 0, 0)
    secs = int((b-a).total_seconds())
    name = str(secs) + ".log"

    logger = logging.getLogger(name)
    logger.setLevel(logging_level)
    fh = logging.FileHandler(join(log_dir, name))
    fh.setLevel(logging_level)
    ch = logging.StreamHandler()
    ch.setLevel(logging_level)
    # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    formatter = logging.Formatter('[%(name)s: %(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
                                  '%m-%d %H:%M:%S')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.propagate = True
    return logger
