import datetime
import os
import logging

UTIL_DIR = os.path.dirname(os.path.realpath(__file__))
PROJ_DIR = os.path.dirname(UTIL_DIR)
EXPT_DIR = os.path.join(PROJ_DIR, 'expts')
STR_TIME_FORMAT = '%Y-%m-%d-%H%M'

def setup_logging(args):
    ##########
    # Logging
    ##########
    # File in which to log output.
    # Just import logging, config in a library to log to the same location.
    if not os.path.exists(EXPT_DIR):
        os.mkdir(EXPT_DIR)
    datetime_stamp = datetime.datetime.now().strftime(STR_TIME_FORMAT)
    args.log_dir = os.path.join(EXPT_DIR, datetime_stamp)
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    info_logfile = os.path.join(args.log_dir, 'log.INFO') 
    debug_logfile = os.path.join(args.log_dir, 'log.DEBUG')

    try:
        info_f = open(info_logfile, 'r')
        debug_f = open(debug_logfile, 'r')
    except IOError:
        info_f = open(info_logfile, 'w')
        debug_f = open(debug_logfile, 'w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s] %(message)s')

    # Set up a streaming logger.
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    fh_info = logging.FileHandler(info_logfile)
    fh_info.setLevel(logging.INFO)
    fh_info.setFormatter(formatter)

    fh_debug = logging.FileHandler(debug_logfile)
    fh_debug.setLevel(logging.DEBUG)
    fh_debug.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh_info)
    logger.addHandler(fh_debug)
