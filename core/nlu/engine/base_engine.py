import logging, os
from config import ROOT_DIR
logging.disable(logging.WARNING)

import tensorflow as tf
from ..tools.utils import DatasetLoader, space_punct
from pathlib import Path

class NLUEngine():
    def __init__(self):
        # TODO
        pass

    def input_preprocess(self, text):
        return space_punct(text)

    def __call__(self, text):
        """
        Uses classificier and tagger to classify intents and extract tags from seq.
        """
        text = self.input_preprocess(text)
        inputs = tf.constant(self.featurizer.encode(text))[None, :]  # batch_size = 1

        intent_class = self.classifier.classify(inputs) # id of intent class
        # tag_logits = self.tagger.tag(inputs)
        # tag_ids = tag_logits.numpy().argmax(axis=-1)[0, 1:-1] # logits of tags

        # return self.decode_predictions(text, intent_id, tag_ids)
        return intent_class

    def decode_predictions(self, text, intent_id, tag_ids):
        """
        Tagger and classifier output to json-like data
        {'intent' : name, 'tags' : {'a' : 'b'}}
        """
        info = {"intent": intent_id}
        collected_tags = {}
        active_tag_words = []
        active_tag_name = None
        #   collect all tags from output
        for word in text.split():
            tokens = self.tokenizer.tokenize(word)
            current_word_tag_ids = tag_ids[:len(tokens)]
            tag_ids = tag_ids[len(tokens):]
            current_word_tag_name = self.id2tag[current_word_tag_ids[0]]

            # print(current_word_tag_name)

            if current_word_tag_name == "O":
                if active_tag_name: # sequence of tags separated with non-tag
                    active_tag_name, active_tag_words = None, []
                # else: start of sentence without any tags
            else:
                tag_name = current_word_tag_name[2:]
                if active_tag_name is None or active_tag_name != tag_name: # new tag
                    if tag_name in collected_tags.keys():
                        collected_tags[tag_name].append(word)
                    else:
                        collected_tags.update({tag_name : [word]})
                    active_tag_name = tag_name
                elif active_tag_name == tag_name: # I-tag in sequence of tags
                    collected_tags[tag_name][-1] += ' ' + word

        info["tags"] = collected_tags
        return info