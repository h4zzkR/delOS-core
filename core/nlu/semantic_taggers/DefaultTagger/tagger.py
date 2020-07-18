import os
from config import ROOT_DIR
import tensorflow as tf
from transformers import BertTokenizer
from pathlib import Path
from .bert_model import BertSemanticTagsExtractor as Tagger

ROOT_DIR = Path(__file__).parent.absolute()


class SemanticTagsExtractor():
    def __init__(self, model_name, id2tag, load=True):

        # self.tokenizer = tokenizer
        self.id2tag = id2tag
        # slot_names, slot_map = load_slots_map(ROOT_DIR)
        tags_number = len(self.id2tag.keys())
        self.model = Tagger(model_name, tags_number)

        ckp_path = self.check_model()
        self.model.load_weights(ckp_path)


    def check_model(self):
        p = os.path.join(ROOT_DIR, 'model/')
        ckp = Path(os.path.join(p, 'checkpoint')).read_text()
        m = ckp.splitlines()[0].split(': ')[-1]
        p = os.path.join(p, m[1:-1])
        return p

    def tag(self, inputs):
        return self.model(inputs)


if __name__ == "__main__":
    model = SemanticTagsExtractor()
    inp = input()
    print(model.tag(inp))