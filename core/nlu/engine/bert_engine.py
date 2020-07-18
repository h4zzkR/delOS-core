from .base_engine import NLUEngine
import tensorflow as tf
from pathlib import Path

from ..tools.utils import DatasetLoader, space_punct
from ..intent_classifier.classifier import RequestIntentClassifier
from ..semantic_taggers.DefaultTagger.tagger import SemanticTagsExtractor

class NLUBertEngine(NLUEngine):
    def __init__(self, model_name="bert-base-cased"):
        super().__init__(model_name)
        d = DatasetLoader('merged') # TODO add here dataset param or join all dsets
        self.intent2id, self.id2intent = d.load_intents_map()
        self.tag2id, self.id2tag = d.load_tags_map()

        self.classifier = RequestIntentClassifier(model_name, self.id2intent)
        self.tagger = SemanticTagsExtractor(model_name, self.id2tag)

if __name__ == "__main__":
    nlu = NLUBertEngine()
    while True:
        text = input()
        print(nlu(text))