import os
from pathlib import Path

# package
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(ROOT_DIR, 'configuration.conf')
PACKAGE_NAME = "delOS"
DATA_PACKAGE_NAME = "data"
DATA_PATH = os.path.join(ROOT_DIR, DATA_PACKAGE_NAME)

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