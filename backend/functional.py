import os, sys, json
import tensorflow as tf
import datetime
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def read_config():
    with open(os.path.join(ROOT_DIR, 'config.json')) as f:
        data = json.load(f)

    return data
    
def tf_set_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

def elapsed_time(start):
    now = datetime.datetime.now()
    time = now - start
    return time.seconds

def dump(dictionary, path, **kwargs):
    if ROOT_DIR not in path:
        path = os.path.join(ROOT_DIR, path)
    with open(path, 'w') as outfile:
        json.dump(dictionary, outfile, **kwargs)

def jsonread(path):
    with open(path, "r") as read_file:
        data = json.load(read_file)
    return data

def listdir(path):
    return next(os.walk(path))[1]

