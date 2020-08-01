import pandas as pd
import numpy as np
import os
import json
import re
from pathlib import Path
from backend.config import ROOT_DIR, NLU_CONFIG, IMAP_PATH

# TODO: вынести некоторые функции в root_dir config.py


def space_punct(text):
    """
    Hey,is it question? -> Hey , is it question ?
    """
    return re.sub('(?<! )(?=[.,!?()])|(?<=[.,!?()])(?! )', r' ', text)

def dump(dictionary, path, **kwargs):
    if ROOT_DIR not in path:
        path = os.path.join(ROOT_DIR, path)
    with open(path, 'w') as outfile:
        json.dump(dictionary, outfile, **kwargs)

def jsonread(path):
    with open(path, "r") as read_file:
        data = json.load(read_file)
    return data

def listdir(path):
    return next(os.walk(path))[1]


# TOOLS FOR STANDART DATASETS

def encode_token_labels(text_sequences, slot_names, featurizer, slot_map, max_length=NLU_CONFIG['max_seq_length']+2):
    encoded = np.zeros(shape=(len(text_sequences), max_length), dtype=np.int32)
    for i, (text_sequence, word_labels) in enumerate(zip(text_sequences, slot_names)):
        encoded_labels = []
        for word, word_label in zip(text_sequence.split(), word_labels.split()):
            tokens = featurizer.tokenize(word)
            encoded_labels.append(slot_map[word_label])
            expand_label = word_label.replace("B-", "I-")
            if not expand_label in slot_map:
                expand_label = word_label
            encoded_labels.extend([slot_map[expand_label]] * (len(tokens) - 1))
        encoded[i, 1:len(encoded_labels) + 1] = encoded_labels
    return encoded

def encode_dataset(featurizer, text_sequences, max_length=NLU_CONFIG['max_seq_length']):
    token_ids = np.zeros(shape=(len(text_sequences), max_length),
                         dtype=np.int32)
    for i, text_sequence in enumerate(text_sequences):
        encoded = featurizer.tokenize(text_sequence)
        token_ids[i, 0:len(encoded)] = encoded
    return token_ids


class DatasetLoader():
    def __init__(self, dset_intents_name, dset_tags_name=None):
        """
        By default intents dataset is equal to tags dataset
        You can set different datasets by specify datasets name
        """
        self.ds_equal = False
        if dset_tags_name is None:
            dset_tags_name = dset_intents_name
            self.ds_equal = True

        self.intents_name = dset_intents_name
        self.tags_name = dset_tags_name

        self.dset_intents_dir = os.path.join(ROOT_DIR, self.intents_name)
        self.dset_tags_dir = os.path.join(ROOT_DIR, self.tags_name)

    def load_intents_map(self, dset_name=None):
        if dset_name is not None:
            self.dset_intents_dir = os.path.join(ROOT_DIR, dset_name)
        intent_names = Path(os.path.join(self.dset_intents_dir, "vocab.intent")).read_text().split()
        intent2id = dict((label, idx) for idx, label in enumerate(intent_names))
        id2intent = {intent2id[i] : i for i in intent2id.keys()}
        return intent2id, id2intent

    def load_tags_map(self, dset_name=None):
        if dset_name is not None:
            self.dset_tags_dir = os.path.join(ROOT_DIR, dset_name)
        tag_names = Path(os.path.join(self.dset_tags_dir, "vocab.tag")).read_text().strip().splitlines()
        tag_map = {}
        tag2id = dict((label, idx) for idx, label in enumerate(tag_names))
        id2tag = {tag2id[i] : i for i in tag2id.keys()}
        return tag2id, id2tag


    def parse_line(self, line):
        """
        DO NOT USE IT
        Parse this like:

        'Add:O Don:B-entity_name and:I-entity_name Sherri:I-entity_name to:O 
        my:B-playlist_owner Meditate:B-playlist to:I-playlist Sounds:I-playlist
        of:I-playlist Nature:I-playlist playlist:O <=> AddToPlaylist'

        to:
        
        {'intent_label': 'AddToPlaylist',
        'length': xx
        'word_labels': 'O B-entity_name I-entity_name I-entity_name O',
        'words': 'Add Don and Sherri to my Meditate to Sounds of Nature playlist'}
        """
        utterance_data, intent_label = line.split(" <=> ")
        items = utterance_data.split()
        words = [item.rsplit(":", 1)[0]for item in items]
        word_labels = [item.rsplit(":", 1)[1]for item in items]
        return {
            "intent_label": intent_label,
            "words": " ".join(words),
            "word_labels": " ".join(word_labels),
            "length": len(words),
        }

class IntentsDatasetLoader(DatasetLoader):
    def __init__(self, dset_intents_name, dset_tags_name=None):
        super().__init__(dset_intents_name, dset_tags_name=dset_tags_name)

    def load_prepare_dataset(self):
        print('Loading data...')

        intent2id, id2intent = self.load_intents_map()

        df_train = pd.read_csv(os.path.join(self.dset_intents_dir, 'train.csv'))
        y_train = df_train["intent_label"].map(intent2id).values
        try:
            df_valid = pd.read_csv(os.path.join(self.dset_intents_dir, 'valid.csv'))
            y_valid = df_valid["intent_label"].map(intent2id).values
        except FileNotFoundError:
            df_valid = None
            y_valid = None
        
        return df_train, y_train, df_valid, y_valid, intent2id, id2intent, None, None

class TagsDatasetLoader(DatasetLoader):
    def __init__(self, dset_intents_name, dset_tags_name=None):
        super().__init__(dset_intents_name, dset_tags_name=dset_tags_name)

    def load_prepare_dataset(self):
        print('Loading data...')

        tag2id, id2tag = self.load_tags_map()

        df_train = pd.read_csv(os.path.join(self.dset_tags_dir, 'train.csv'))
        try:
            df_valid = pd.read_csv(os.path.join(self.dset_tags_dir, 'valid.csv'))
        except FileNotFoundError:
            df_valid = None
        
        return df_train, df_valid, None, None, tag2id, id2tag

def load_intents_map(intents_dset_name):
    d = DatasetLoader(dset_intents_name=intents_dset_name)
    return d.load_intents_map()

def load_tags_map(tags_dset_name):
    d = DatasetLoader(dset_intents_name=tags_dset_name)
    return d.load_tags_map()

def load_map(self, text_vocab_path):
    """
    Path to the vocab.__entity_name__
    This file located at dataset dir
    """
    if dset_name is not None:
        path = os.path.join(ROOT_DIR, text_vocab_path)
    names = Path(path).read_text().strip().splitlines()
    ent_map = {}
    ent2id = dict((label, idx) for idx, label in enumerate(names))
    id2ent = {ent2id[i] : i for i in ent2id.keys()}
    return ent2id, id2ent

# END OF TOOLS FOR STANDART DATASETS

def search_all_intents():
    print(ROOT_DIR)
    path = os.path.join(ROOT_DIR, 'data/nlu_data/')
    dsets = next(os.walk(path))[1]
    dsets = [d for d in dsets if '#' not in d]
    intents_vocab_list = []
    for directory in dsets:
        vocab_file_path = os.path.join(path, directory, 'vocab.intent')
        intents_vocab = open(vocab_file_path, 'r').readlines()
        for i in intents_vocab:
            if i not in intents_vocab_list:
                intents_vocab_list.append(i)
            # else:
                print('Found dublicate of', i)
    with open(os.path.join(ROOT_DIR, 'core/nlu/intent_manager/vocab.intent'), 'w') as file:
        for i in intents_vocab_list:
            if '\n' not in i:
                i += '\n'
            file.write(i)

def get_all_intents():
    intents_vocab_list = []
    with open(os.path.join(ROOT_DIR, 'core/nlu/intent_manager/vocab.intent'), 'r') as file:
        for i in file.readlines():
            intents_vocab_list.append(i.replace('\n', ''))
    return intents_vocab_list
    
def map_all_intents():
    ilist = get_all_intents()
    imap = {}
    for i in ilist:
        imap.update({i : ['BuiltinTagger', 'DefaultTagger']})
    dump(imap, 'core/nlu/intent_manager/taggers_map.json', indent=4, sort_keys=True)

def get_intents_map():
    return jsonread(IMAP_PATH)