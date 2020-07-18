import os
from config import ROOT_DIR
import tensorflow as tf
from transformers import BertTokenizer
from pathlib import Path
from .bert_model import BertClassifierModel as Classifier

ROOT_DIR = Path(__file__).parent.absolute()


# not nearest future: add configs for various models

class RequestIntentClassifier():
    def __init__(self, model_name, id2intent, load=True):

        self.id2intent = id2intent
        intents_number = len(self.id2intent.keys())

        self.model = Classifier(model_name, intents_number)

        ckp_path = self.check_model()
        self.model.load_weights(ckp_path)

    def check_model(self):
        p = os.path.join(ROOT_DIR, 'model/')
        ckp = Path(os.path.join(p, 'checkpoint')).read_text()
        m = ckp.splitlines()[0].split(': ')[-1]
        p = os.path.join(p, m[1:-1])
        return p

    def classify(self, inputs):
        class_id = self.model(inputs).numpy().argmax(axis=1)[0]
        # return class_id
        return self.id2intent[class_id]


if __name__ == "__main__":
    from ..tools.utils import DatasetLoader, encode_dataset, encode_token_labels
    d = DatasetLoader('data/nlu_data/custom')
    df_train, df_valid, intent2id, id2intent, tag2id, id2tag = d.load_prepare_dataset()
    model = RequestIntentClassifier("bert-base-cased", id2intent)

    while True:
        inp = input()
        tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        inp = tf.constant(tokenizer.encode(inp))[None, :]
        print(model.classify(inp))