import logging, os
import importlib
import tensorflow as tf
import numpy as np
import importlib
logging.disable(logging.WARNING)

from backend.config import ROOT_DIR, NLU_CONFIG, TAGGERS_DIR

from backend.core.nlu.tools.utils import load_intents_map, space_punct, \
        encode_dataset, encode_token_labels, get_intents_map, listdir
from pathlib import Pathx

from ..featurizers.transformer_featurizer import SentenceFeaturizer
from ..intent_classifier.classifier import IntentClassifier
from backend.core.nlu.semantic_tagger.builtin_entities_model import BuiltinEntityTagger
from ..semantic_tagger.tagger import SemanticTagsExtractor as DefaultTagger


class NLUEngine():
    def __init__(self, _default_models = ['BuiltinTagger']):
        super().__init__(_default_models)

        self.intent2id, self.id2intent = load_intents_map(NLU_CONFIG['intents_classificier_dataset'])
        self.featurizer = SentenceFeaturizer()
        self.classifier = IntentClassifier(self.id2intent, NLU_CONFIG['classifier_model'])

        self.sem_taggers = dict()
        self.input_processor = space_punct

        # self.builtin_tagger = BuiltinEntityTagger()
        
        # TODO Edit it

    @property       
    def fitted(self):
        """Whether or not the intent classifier and taggers has already been fitted"""
        pass

    def preprocess_text(self, text):
        return self.featurizer.featurize(self.input_processor(text))

    def parse(self, text, intents=None, top_n=None):
        """Performs intent parsing on the provided *text* by first classifying
        the intent and then using the correspond slot filler to extract slots
        Args:
            text (str): input
            intents (str or list of str): if provided, reduces the scope of
                intent parsing to the provided list of intents
            top_n (int, optional): when provided, this method will return a
                list of at most top_n most likely intents, instead of a single
                parsing result.
                Note that the returned list can contain less than ``top_n``
                elements, for instance when the parameter ``intents`` is not
                None, or when ``top_n`` is greater than the total number of
                intents.
        Returns:
            dict or list: the most likely intent(s) along with the extracted
            slots. See :func:`.parsing_result` and :func:`.extraction_result`
            for the output format.
        Raises:
            NotTrained: when the intent parser is not fitted
        """
        enc_seq, pool_out = self.preprocess_text(text)
        intent_class, probs = self.classifier.classify(pool_out)

        builtin_tags = self.builtin_tagger.tag(text)
        # out_tags = self.taggers_graph(enc_seq, intent_class)
        out_tags = self.default_tagger.tag(enc_seq)
        print(out_tags)
        input()
        # TODO: decode out in tagger class

        return self.decode_predictions(text, intent_class, probs, tag_ids)

        


