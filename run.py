from pathlib import Path
curdir = Path(__file__).parent.absolute()
import sys, os; sys.path.insert(0, str(curdir))

import tensorflow as tf
from core.nlu.intent_classificator.base import SlotIntentDetectorModelBase