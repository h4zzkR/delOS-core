import os
from config import ROOT_DIR
import tensorflow as tf
import numpy as np
from pathlib import Path
from .dense_model import DenseClassifierModel as Classifier

# not nearest future: add configs for various models

class IntentClassifier():
    def __init__(self, id2intent, model_name='dense', load=True):

        self.id2intent = id2intent
        intents_number = len(self.id2intent.keys())

        self.model = Classifier(intents_number)

        ckp_path = self.check_model()
        self.model.load_weights(ckp_path)

    def check_model(self):
        p = os.path.join(ROOT_DIR, 'core/nlu/intent_classifier/model/')
        ckp = Path(os.path.join(p, 'checkpoint')).read_text()
        m = ckp.splitlines()[0].split(': ')[-1]
        p = os.path.join(p, m[1:-1])
        return p

    def classify(self, inputs):
        out = self.model(inputs)
        out = np.squeeze(np.array(out))
        class_id = np.argmax(out)
        return self.id2intent[class_id]


if __name__ == "__main__":
    pass
    # from ..tools.utils import DatasetLoader, encode_dataset, encode_token_labels
    # d = DatasetLoader('data/nlu_data/custom')
    # df_train, df_valid, intent2id, id2intent, tag2id, id2tag = d.load_prepare_dataset()
    # model = RequestIntentClassifier("bert-base-cased", id2intent)

    # while True:
    #     inp = input()
    #     tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    #     inp = tf.constant(tokenizer.encode(inp))[None, :]
    #     print(model.classify(inp))