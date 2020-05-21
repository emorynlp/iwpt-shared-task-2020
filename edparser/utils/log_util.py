# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-08-24 22:12
import datetime
import logging
import os
import sys


def init_logger(name=None, root_dir=None, level=logging.INFO, mode='w') -> logging.Logger:
    if not name:
        name = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    logFormatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
    rootLogger = logging.getLogger(os.path.join(root_dir, name) if root_dir else name)
    rootLogger.propagate = False

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    attached_to_std = False
    for handler in rootLogger.handlers:
        if isinstance(handler, logging.StreamHandler):
            if handler.stream == sys.stderr or handler.stream == sys.stdout:
                attached_to_std = True
                break
    if not attached_to_std:
        rootLogger.addHandler(consoleHandler)
    rootLogger.setLevel(level)
    consoleHandler.setLevel(level)

    if root_dir:
        os.makedirs(root_dir, exist_ok=True)
        fileHandler = logging.FileHandler("{0}/{1}.log".format(root_dir, name), mode=mode)
        fileHandler.setFormatter(logFormatter)
        rootLogger.addHandler(fileHandler)
        fileHandler.setLevel(level)

    return rootLogger


def set_tf_loglevel(level=logging.ERROR):
    if level >= logging.FATAL:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
    if level >= logging.ERROR:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '2'
    if level >= logging.WARNING:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '1'
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
        os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '0'
    shut_up_python_logging()
    logging.getLogger('tensorflow').setLevel(level)


def shut_up_python_logging():
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    import absl.logging
    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False


logger = init_logger(name='hanlp', level=os.environ.get('HANLP_LOG_LEVEL', 'INFO'))


# shut_up_python_logging()

# shut up tensorflow
# set_tf_loglevel()


def enable_debug(debug=True):
    logger.setLevel(logging.DEBUG if debug else logging.ERROR)
