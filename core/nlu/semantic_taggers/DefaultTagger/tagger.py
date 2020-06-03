import os
import tensorflow as tf
from transformers import BertTokenizer
from pathlib import Path
from .bert_model import BertSemanticTagsExtractor as Tagger


class SemanticTagsExtractor():
    def __init__(self, model_name, id2tag, load=True):

        curdir = Path(__file__).parent.absolute()
        # self.tokenizer = tokenizer
        self.id2tag = id2tag
        # slot_names, slot_map = load_slots_map(curdir)
        tags_number = len(self.id2tag.keys())
        self.model = Tagger(model_name, tags_number)

        ckp_path = self.check_model(curdir)
        self.model.load_weights(ckp_path)


    def check_model(self, curdir):
        p = os.path.join(curdir, 'model/')
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