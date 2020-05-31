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



if __name__ == "__main__":
    import logging, os
    logging.disable(logging.WARNING)
    from transformers import TFBertModel, BertTokenizer
    from ..nlu_data.utils import DatasetLoader
    from ..semantic_taggers.tagger.model import SemanticTagsExtractor


    d = DatasetLoader('merged')
    intent2id, id2intent = d.load_intents_map()
    tag2id, id2tag = d.load_tags_map()
    bertbase = TFBertModel.from_pretrained('bert-base-cased')
    # model = RequestIntentClassifier('bert-base-cased', id2intent)
    semtag = SemanticTagsExtractor(bertbase, id2tag)
    model = RequestIntentClassifier(bertbase, id2intent)
    
    # Tell me the weather forecast for here
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    while True:
        text = input()
        inp = tf.constant(tokenizer.encode(text))[None, :]  # batch_size = 1
        print(model.classify(inp))