import os
from termcolor import colored
from core.callback.logger import init_logger

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(ROOT_DIR, 'core/callback/logs')

LOGGER = init_logger(__name__, testing_mode=False, dir=ROOT_DIR)