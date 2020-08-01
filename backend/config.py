import os
from pathlib import Path
from backend.functional import read_config

# DIR&PATH
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

CORE_DIR = os.path.join(ROOT_DIR, 'core')
NLU_DIR = os.path.join(CORE_DIR, 'nlu')
TAGGER_DIR = os.path.join(NLU_DIR, 'semantic_tagger')
IMAP_PATH = os.path.join(NLU_DIR, 'tag_intent_manager/taggers_map.json')

CONFIG_DIR = os.path.join(ROOT_DIR, 'config.py')
PACKAGE_NAME = "delOS"
DATA_PACKAGE_NAME = "data"
DATA_PATH = os.path.join(ROOT_DIR, DATA_PACKAGE_NAME)

CONFIG = read_config()
NLU_CONFIG = CONFIG['nlu_engine']
NER_CONFIG = CONFIG['ner_engine']

# MODELLING
MODELS_PARAMS = os.path.join(ROOT_DIR, 'configs/model_params.json')

# miscellaneous
BUILTIN_ENTITY_TAGGER = "builtin_entity_tagger"
INTENT = "intent"
INTENTS = "intents"
ENTITIES = "entities"
ENTITY = "entity"
ENTITY_KIND = "entity_kind"


AUTOMATICALLY_EXTENSIBLE = "automatically_extensible"
USE_SYNONYMS = "use_synonyms"
SYNONYMS = "synonyms"
DATA = "data"
RESOLVED_VALUE = "resolved_value"
SLOT_NAME = "slot_name"
TEXT = "text"
UTTERANCES = "utterances"
LANGUAGE = "language"
VALUE = "value"
NGRAM = "ngram"
TOKEN_INDEXES = "token_indexes"
CAPITALIZE = "capitalize"
UNKNOWNWORD = "unknownword"
VALIDATED = "validated"
START = "start"
END = "end"
CUSTOM_ENTITY_PARSER = "custom_entity_parser"
MATCHING_STRICTNESS = "matching_strictness"
RANDOM_STATE = "random_state"

# resources
RESOURCES = "resources"
METADATA = "metadata"
STOP_WORDS = "stop_words"
NOISE = "noise"
GAZETTEERS = "gazetteers"
STEMS = "stems"
CUSTOM_ENTITY_PARSER_USAGE = "custom_entity_parser_usage"
WORD_CLUSTERS = "word_clusters"
GAZETTEER_ENTITIES = "gazetteer_entities"

# builtin entities TODO: REDO
SNIPS_AMOUNT_OF_MONEY = "snips/amountOfMoney"
SNIPS_DATETIME = "snips/datetime"
SNIPS_DURATION = "snips/duration"
SNIPS_NUMBER = "snips/number"
SNIPS_ORDINAL = "snips/ordinal"
SNIPS_PERCENTAGE = "snips/percentage"
SNIPS_TEMPERATURE = "snips/temperature"
