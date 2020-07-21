import os
from pathlib import Path
from functional import read_config

# package
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.join(ROOT_DIR, 'core')
NLU_DIR = os.path.join(CORE_DIR, 'nlu')

CONFIG_DIR = os.path.join(ROOT_DIR, 'config.py')
PACKAGE_NAME = "delOS"
DATA_PACKAGE_NAME = "data"
DATA_PATH = os.path.join(ROOT_DIR, DATA_PACKAGE_NAME)

CONFIG = read_config()
NLU_CONFIG = CONFIG['nlu_engine']
NER_CONFIG = CONFIG['ner_engine']

# miscellaneous
AUTOMATICALLY_EXTENSIBLE = "automatically_extensible"
USE_SYNONYMS = "use_synonyms"
SYNONYMS = "synonyms"
DATA = "data"
INTENTS = "intents"
ENTITIES = "entities"
ENTITY = "entity"
ENTITY_KIND = "entity_kind"
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
BUILTIN_ENTITY_PARSER = "builtin_entity_parser"
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

# builtin entities
SNIPS_AMOUNT_OF_MONEY = "snips/amountOfMoney"
SNIPS_DATETIME = "snips/datetime"
SNIPS_DURATION = "snips/duration"
SNIPS_NUMBER = "snips/number"
SNIPS_ORDINAL = "snips/ordinal"
SNIPS_PERCENTAGE = "snips/percentage"
SNIPS_TEMPERATURE = "snips/temperature"
