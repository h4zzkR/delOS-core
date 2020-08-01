import logging, os
import importlib
import tensorflow as tf
import numpy as np
logging.disable(logging.WARNING)

from config import ROOT_DIR, NLU_CONFIG, TAGGERS_DIR

from ..tools.utils import load_intents_map, space_punct, \
        encode_dataset, encode_token_labels, get_intents_map, listdir
from pathlib import Path

from ..semantic_taggers.DefaultTagger.tagger import SemanticTagsExtractor
from ..semantic_taggers.BuiltinTagger.tagger import BuiltinEntityTagger

class TaggerManager():
    def __init__(self, default_models=['BuiltinTagger']):
        self._default_models = default_models
        self._tag_modules = []
        self.tag_modules = {}
        
        self._load_intent2tagger()
        self._load_tag_modules()
    
    def _load_intent2tagger(self):
        self.intent2tagger = get_intents_map()

    def _load_tag_modules(self):
        imap = get_intents_map()
        for key in imap.keys():
            self._tag_modules += [im for im in imap[key] if im not in self._default_models]
        self._tag_modules = list(set(self._tag_modules)) # drop dublicates
        # self._load_modules()

    # def _load_modules(self):
    #     for m in self._tag_modules:
    #         self.tag_modules.update({
    #             m : importlib.import_module(f"core.nlu.semantic_taggers.{m}.tagger").SemanticTagsExtractor()
    #             # m : SemanticTagsExtractor()
    #         })
    #         print(f"{m} module loaded, code {self.tag_modules[m].check_loaded()}")

    def taggers_graph(self, inputs, intent_class):
        tags = {}
        for module in self.intent2tagger[intent_class]:
            if module not in self._default_models:
                out = self.tag_modules[module].tag(inputs)
                tags.update({module : out})
        return tags


    def get_all_tagger_modules(self):
        return self._default_models + list(self._tag_modules.keys())

    def get_intent_tagger_modules(self):
        return list(self._tag_modules.keys())
        
