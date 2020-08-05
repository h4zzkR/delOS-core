import logging, os
import importlib
import tensorflow as tf
import numpy as np
import importlib
logging.disable(logging.WARNING)

from backend.config import ROOT_DIR, NLU_CONFIG, TAGGERS_DIR
from .base_engine_abstract import NLUEngine

from backend.core.nlu.tools.utils import load_intents_map, space_punct, \
        encode_dataset, encode_token_labels, get_intents_map, listdir
from pathlib import Pathx

from ..featurizers.transformer_featurizer import SentenceFeaturizer
from ..intent_classifier.classifier import IntentClassifier
from backend.core.nlu.semantic_tagger.builtin_entities_model import BuiltinEntityTagger
from ..semantic_tagger.tagger import SemanticTagsExtractor as DefaultTagger


class ProbabilisticNLUEngine(NLUEngine):
    def __init__(self, _default_models = ['BuiltinTagger']):
        super().__init__()
        self.intent_classifier = None
        self.taggers = dict()

    def _load_modules(self):
        self.featurizer = SentenceFeaturizer()
        self.input_processor = space_punct

    def encode_text(self, text):
        return self.featurizer.featurize(self.input_processos(text))

    def fit(self, dataset, force_retrain=False):
        if self.intent_classifier is None:
            self.intent_classifier = None #redo