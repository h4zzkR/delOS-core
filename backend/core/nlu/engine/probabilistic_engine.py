import logging
import numpy as np
import datetime
logging.disable(logging.WARNING)

from backend.configuration.config import NLU_CONFIG
from .base_engine_abstract import NLUEngine
from backend.functional import elapsed_time

from backend.core.nlu.tools.utils import space_punct, \
    get_all_intents, \
        load_id2tag, load_id2intent
    

from backend.core.nlu.intent_classifier.classifier import IntentClassifier
from backend.core.nlu.featurizers.transformer_featurizer import SentenceFeaturizer
from backend.core.nlu.semantic_tagger.builtin_entities_model import BuiltinEntityTagger
from backend.core.nlu.semantic_tagger.tagger import SemanticTagger


class ProbabilisticNLUEngine(NLUEngine):
    def __init__(self):
        super().__init__()
        self.intent_classifier = None
        self.taggers = dict()
        self.dataset = None

    def _load_modules(self):
        self.featurizer = SentenceFeaturizer()
        # input for first initial loading
        self.featurizer.featurize('eval')
        # LOGGING
        print(f"Featurizer module loaded")
        self.input_processor = space_punct

    def eval(self, dataset=NLU_CONFIG['universal_dataset']):
        self._load_modules()
        if self.dataset:
            dataset = self.dataset
        self.id2intent = load_id2intent(dataset)
        self.id2tag = load_id2tag(dataset)
        self.builtin_tagger = BuiltinEntityTagger()

    def encode_text(self, text):
        return self.featurizer.featurize(self.input_processor(text))

    def fit(self, dataset=NLU_CONFIG["universal_dataset"], force_retrain=False, merge_taggers=None):
        self.dataset = dataset
        intents = get_all_intents(dataset)
        if self.intent_classifier is None:
            self.intent_classifier = IntentClassifier()
            self.intent_classifier.load()
        if force_retrain or not self.intent_classifier.fitted:
            self.intent_classifier.fit(dataset)
        start_time = datetime.datetime.now()
        for intent_name in intents:
            if self.taggers.get(intent_name) is None:
                self.taggers[intent_name] = SemanticTagger()
                self.taggers[intent_name].load(intent_name)
            if merge_taggers and intent_name in merge_taggers:
                if force_retrain or not self.taggers[intent_name].fitted():
                    self.taggers[intent_name].fit(dataset, intent_name, merge_taggers.split())
            else:
                if force_retrain or not self.taggers[intent_name].fitted():
                    self.taggers[intent_name].fit(dataset, intent_name, merge_taggers)
        # LOGGING
        print(f'Fitted tagger models in {elapsed_time(start_time)}s')
        return self

    def mixin_builtin_tags(self, result, builtin_tags):
        if builtin_tags is not None:
            result['tags'].update(builtin_tags)
        return result

    def top_n_intent_names(self, probs):
        return np.argsort(probs)

    def parse(self, text, top_n=None, intents=None):
        if isinstance(intents, str):
            intents = {intents}
        elif isinstance(intents, list):
            intents = list(intents)

        tokenized_seq, enc_seq, pool_out = self.encode_text(text)
        pool_out = self.featurizer.encode(self.input_processor(text))
        builtin_tags = self.builtin_tagger.tag(text)
        # print(self.builtin_tagger.tag('buy milk for 2.5 dollars')); sys.exit()
        if len(builtin_tags) == 0:
            builtin_tags = None

        if top_n is None:
            intent_id, probs = self.intent_classifier.classify(pool_out)
            intent_name = self.id2intent[intent_id]
            if intent_name is not None:
                tags_logits = self.taggers[intent_name].tag(enc_seq)
            else:
                tags_logits = None
            # print(tags_logits)
            return self.mixin_builtin_tags(self.build_result(text, intent_name, tags_logits), \
                builtin_tags)

        results = []
        _, probs = self.intent_classifier.classify(pool_out)
        intent_ids = self.top_n_intent_names(probs)
        for intent_id in intent_ids[-1:len(intent_ids)-top_n-1]:
            intent_name = self.id2intent[intent_id]
            if intent_name is not None:
                tags_logits = self.taggers[intent_name].tag(enc_seq)
            else:
                tags_logits = []
            results.append(self.build_result(text, intent_name, tags_logits))
        return results

    def build_result(self, text, intent_name, tag_logits):
        info = {"intent": intent_name}
        collected_tags = {}
        active_tag_words = []
        active_tag_name = None
        #   collect all tags from output
        for word in text.split():
            tokens = self.featurizer.tokenize(word)
            current_word_tag_logits = tag_logits[:len(tokens)]
            tag_logits = tag_logits[len(tokens):]
            current_word_tag_name = self.id2tag[current_word_tag_logits[0]]

            if current_word_tag_name == "O":
                if active_tag_name: # sequence of tags separated with non-tag
                    active_tag_name, active_tag_words = None, []
                # else: start of sentence without any tags
            else:
                tag_name = current_word_tag_name[2:]
                if active_tag_name is None or active_tag_name != tag_name: # new tag
                    if tag_name in collected_tags.keys():
                        collected_tags[tag_name]['value'].append(word)
                    else:
                        collected_tags.update({tag_name : {'value' : [word]}})
                    active_tag_name = tag_name
                elif active_tag_name == tag_name: # I-tag in sequence of tags
                    collected_tags[tag_name]['value'][-1] += ' ' + word

        info["tags"] = collected_tags
        return info

if __name__ == "__main__":
    obj = ProbabilisticNLUEngine()
    obj.fit('data/nlu_data/snipsai', force_retrain=True)#, merge_taggers='turnLightOn turnLightOff')
    obj.eval()
    while True:
        inp = input()
        print(obj.parse(inp))
    # obj.load('intent')