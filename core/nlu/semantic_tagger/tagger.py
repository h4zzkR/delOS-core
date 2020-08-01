import os
from config import ROOT_DIR, NLU_CONFIG
import tensorflow as tf
from transformers import BertTokenizer
from pathlib import Path
import numpy as np
from ...tools.utils import encode_dataset, jsonread, load_tags_map
from .dense_model import DenseTagsExtractor as Tagger


class SemanticTagsExtractor():
    def __init__(self, id2tag=None, load=True, path2model=None):

        # self.tokenizer = tokenizer
        self.id2tag = id2tag
        # slot_names, slot_map = load_slots_map(ROOT_DIR)

        if not path2model: path2model = NLU_CONFIG['sem_tagger_model']
        ckp_path, params = self.check_model(path2model)
        self.dataset_name = params['dataset_name']

        if self.id2tag:
            self.tags_number = len(self.id2tag)
        else:
            self.tags_number = params['output_length']
            self.id2tag = load_tags_map(self.dataset_name)

        self.model = Tagger(self.tags_number)
        self.model.load_weights(ckp_path)


    def check_model(self, path2model):
        p = os.path.join(ROOT_DIR, path2model)
        ckp = Path(os.path.join(p, 'checkpoint')).read_text()
        m = ckp.splitlines()[0].split(': ')[-1]
        params = jsonread(os.path.join(p, 'params.json'))
        p = os.path.join(p, m[1:-1])
        return p, params

    def tag(self, inputs):
        out = np.squeeze(np.squeeze(self.model(inputs), 0), 0)
        out = np.argmax(out, 1)[1:-1]
        return out

    def check_loaded(self):
        return 1


if __name__ == "__main__":
    model = SemanticTagsExtractor()
    inp = input()
    print(model.tag(inp))