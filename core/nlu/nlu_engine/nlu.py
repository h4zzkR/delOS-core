import logging, os
logging.disable(logging.WARNING)

import tensorflow as tf
from pathlib import Path
from ..intent_classifier.model import RequestIntentClassifier
from ..semantic_taggers.tagger.model import SemanticTagsExtractor
from ..nlu_data.utils import DatasetLoader, space_punct
from transformers import TFBertModel, BertTokenizer

class IntentManager():
    def __init__(self):
        pass

    def __call__(self, intent, request):
        pass
            

class NLU():
    def __init__(self, model_name="bert-base-cased"):
        self.curdir = Path(__file__).parent.absolute()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        # TFBertBase hugging face transformers' gpu memory usage if efficiently and optimal.
        bertbase = model_name
        d = DatasetLoader('merged') # TODO add here dataset param or join all dsets
        self.intent2id, self.id2intent = d.load_intents_map()
        self.tag2id, self.id2tag = d.load_tags_map()

        self.classifier = RequestIntentClassifier(bertbase, self.id2intent)
        self.tagger = SemanticTagsExtractor(bertbase, self.id2tag)

    def text_prep(self, text):
        return space_punct(text)

    def __call__(self, text):
        text = self.text_prep(text)
        inputs = tf.constant(self.tokenizer.encode(text))[None, :]  # batch_size = 1

        intent_id = self.classifier.classify(inputs)
        tag_logits = self.tagger.tag(inputs)
        tag_ids = tag_logits.numpy().argmax(axis=-1)[0, 1:-1]

        return self.decode_predictions(text, intent_id, tag_ids)

    def decode_predictions(self, text, intent_id, tag_ids):
        # TODO intent_id to intent
        """
        Model output to json-like data
        {'intent' : name, 'tags' : {'a' : 'b'}}
        """
        info = {"intent": intent_id}
        collected_tags = {}
        active_tag_words = []
        active_tag_name = None
        for word in text.split():
            tokens = self.tokenizer.tokenize(word)
            current_word_tag_ids = tag_ids[:len(tokens)]
            tag_ids = tag_ids[len(tokens):]
            current_word_tag_name = self.id2tag[current_word_tag_ids[0]]
            if current_word_tag_name == "O":
                if active_tag_name:
                    collected_tags[active_tag_name] = " ".join(active_tag_words)
                    active_tag_words = []
                    active_tag_name = None
            else:
                # Naive BIO: handling: treat B- and I- the same...
                new_tag_name = current_word_tag_name[2:]
                if active_tag_name is None:
                    active_tag_words.append(word)
                    active_tag_name = new_tag_name
                elif new_tag_name == active_tag_name:
                    active_tag_words.append(word)
                else:
                    collected_tags[active_tag_name] = " ".join(active_tag_words)
                    active_tag_words = [word]
                    active_tag_name = new_tag_name
        if active_tag_name:
            collected_tags[active_tag_name] = " ".join(active_tag_words)
        info["tags"] = collected_tags
        return info


if __name__ == "__main__":
    nlu = NLU()
    while True:
        text = input()
        print(nlu(text))