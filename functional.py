import os, sys, json
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def read_config():
    with open(os.path.join(ROOT_DIR, 'config.json')) as f:
        data = json.load(f)

    return data
    
