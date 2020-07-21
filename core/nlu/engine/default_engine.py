from .base_engine import NLUEngine
import tensorflow as tf
from pathlib import Path
from config import NLU_CONFIG

from ..tools.utils import DatasetLoader, space_punct
from ..featurizers.transformer_featurizer import SentenceFeaturizer
from ..intent_classifier.classifier import IntentClassifier
from ..semantic_taggers.DefaultTagger.tagger import SemanticTagsExtractor

class NLUEngine(NLUEngine):
    def __init__(self):
        super().__init__()
        d = DatasetLoader(NLU_CONFIG['intents_dataset'])
        self.intent2id, self.id2intent = d.load_intents_map()
        self.tag2id, self.id2tag = d.load_tags_map()
        del d

        self.featurizer = SentenceFeaturizer()
        self.classifier = IntentClassifier(self.id2intent, NLU_CONFIG['classifier_model'])
        self.tagger = SemanticTagsExtractor(self.id2tag)

if __name__ == "__main__":
    nlu = NLUEngine()
    # print(nlu('turn off the light in the bedroom'))
    # input()
    while True:
        text = input()
        print(nlu(text))