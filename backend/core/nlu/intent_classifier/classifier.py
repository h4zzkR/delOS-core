import os
from backend.config import ROOT_DIR, NLU_CONFIG
import tensorflow as tf
import numpy as np
from pathlib import Path
from backend.core.nlu.intent_classifier.dense_model import DenseClassifierModel as Classifier
from backend.core.nlu.tools.utils import encode_dataset, jsonread, load_intents_map

# not nearest future: add configs for various models

class IntentClassifier():
    def __init__(self, id2intent=None, load=True, path2model=None):

        self.id2intent = id2intent
        if not path2model: path2model = NLU_CONFIG['classifier_model']
        ckp_path, params = self.check_model(path2model)
        self.dataset_name = params['dataset_name']

        if not self.id2intent:
            self.id2intent = load_intents_map(self.dataset_name)

        self.intents_number = len(self.id2intent)
        self.model = Classifier(self.intents_number)
        self.model.load_weights(ckp_path)

    def check_model(self, path2model):
        p = os.path.join(ROOT_DIR, path2model)
        ckp = Path(os.path.join(p, 'checkpoint')).read_text()
        m = ckp.splitlines()[0].split(': ')[-1]
        params = jsonread(os.path.join(p, 'params.json'))
        p = os.path.join(p, m[1:-1])
        return p, params

    def classify(self, inputs):
        out = self.model(inputs)
        out = np.squeeze(np.array(out))
        class_id = np.argmax(out)
        return self.id2intent[class_id], out


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