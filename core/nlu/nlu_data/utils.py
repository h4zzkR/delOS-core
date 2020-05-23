import pandas as pd
import numpy as np
import os
import re
from pathlib import Path


# PREPROCESS FUNCS

def encode_dataset(tokenizer, text_sequences, max_length):
    """
    Encode sequences with Bert-style tokenizer for inputting
    to the Bert layer.
    """
    token_ids = np.zeros(shape=(len(text_sequences), max_length),
                         dtype=np.int32)
    for i, text_sequence in enumerate(text_sequences):
        encoded = tokenizer.encode(text_sequence)
        token_ids[i, 0:len(encoded)] = encoded
    attention_masks = (token_ids != 0).astype(np.int32)
    return {"input_ids": token_ids, "attention_masks": attention_masks}


def space_punct(text):
    return re.sub('(?<! )(?=[.,!?()])|(?<=[.,!?()])(?! )', r' ', text)



def encode_token_labels(text_sequences, slot_names, tokenizer, slot_map,
                        max_length):
    encoded = np.zeros(shape=(len(text_sequences), max_length), dtype=np.int32)
    for i, (text_sequence, word_labels) in enumerate(
            zip(text_sequences, slot_names)):
        encoded_labels = []
        for word, word_label in zip(text_sequence.split(), word_labels.split()):
            tokens = tokenizer.tokenize(word)
            encoded_labels.append(slot_map[word_label])
            expand_label = word_label.replace("B-", "I-")
            if not expand_label in slot_map:
                expand_label = word_label
            encoded_labels.extend([slot_map[expand_label]] * (len(tokens) - 1))
        encoded[i, 1:len(encoded_labels) + 1] = encoded_labels
    return encoded



class DatasetLoader():
    def __init__(self, dset_name):
        curdir = Path(__file__).parent.absolute()
        self.name = dset_name
        self.dset_dir = os.path.join(curdir, 'dsets', dset_name)

    def load_prepare_dataset(self, dset_name):
        print('Loading data...')
        df_train = pd.read_csv(os.path.join(self.dset_dir, 'train.csv'))
        df_valid = pd.read_csv(os.path.join(self.dset_dir, 'valid.csv'))
        df_test = pd.read_csv(os.path.join(self.dset_dir, 'test.csv'))

        intent2id, id2intent = self.load_intents_map()
        slot_names, slot_map = self.load_slots_map()
        
        return df_train, df_valid, df_test, intent2id, id2intent, slot_map

    def load_intents_map(self):
        intent_names = Path(os.path.join(self.dset_dir, "vocab.intent")).read_text().split()
        intent2id = dict((label, idx) for idx, label in enumerate(intent_names))
        id2intent = {intent2id[i] : i for i in intent2id.keys()}
        return intent2id, id2intent

    def load_slots_map(self):
        slot_names = Path(os.path.join(self.dset_dir, "vocab.slot")).read_text().strip().splitlines()
        slot_map = {}
        for label in slot_names:
            slot_map[label] = len(slot_map)
        return slot_names, slot_map


    def dset2csv(dset_name=None):
        """
        From text file to csv. Run it once
        """
        lines_train = Path(
            os.path.join(self.dset_dir, 'raw/train')
            ).read_text().strip().splitlines()
        lines_valid = Path(
            os.path.join(self.dset_dir, 'raw/valid')
            ).read_text().strip().splitlines()
        lines_test = Path(
            os.path.join(self.dset_dir, 'raw/test')
            ).read_text().strip().splitlines()

        parsed = [self.parse_line(line) for line in lines_train]

        df_train = pd.DataFrame([p for p in parsed if p is not None])
        df_valid = pd.DataFrame([self.parse_line(line) for line in lines_valid])
        df_test = pd.DataFrame([self.parse_line(line) for line in lines_test])

        df_train.to_csv(os.path.join(self.dset_dir, 'train.csv'))
        df_valid.to_csv(os.path.join(self.dset_dir, 'data/valid.csv'))
        df_test.to_csv(os.path.join(self.dset_dir, 'test.csv'))
        # return df_train, df_valid, df_test

    def parse_line(self, line):
        """
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