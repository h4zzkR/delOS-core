import os
import tensorflow as tf
from transformers import BertTokenizer
from pathlib import Path
from .base import RequestIntentClassifier as Classifier


class RequestIntentClassifier():
    def __init__(self, model_name, id2intent, load=True):

        curdir = Path(__file__).parent.absolute()
        self.id2intent = id2intent
        intents_number = len(self.id2intent.keys())

        self.model = Classifier(model_name, intents_number)

        ckp_path = self.check_model(curdir)
        self.model.load_weights(ckp_path)


    def check_model(self, curdir):
        p = os.path.join(curdir, 'model/')
        ckp = Path(os.path.join(p, 'checkpoint')).read_text()
        m = ckp.splitlines()[0].split(': ')[-1]
        p = os.path.join(p, m[1:-1])
        return p

    def classify(self, inputs):
        class_id = self.model(inputs).numpy().argmax(axis=1)[0]
        # return class_id
        return self.id2intent[class_id]
