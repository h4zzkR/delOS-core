import pandas as pd
import numpy as np
import os
import re
from pathlib import Path
from config import ROOT_DIR


def space_punct(text):
    """
    Hey,is it question? -> Hey , is it question ?
    """
    return re.sub('(?<! )(?=[.,!?()])|(?<=[.,!?()])(?! )', r' ', text)

# TOOLS FOR STANDART DATASETS

# def encode_dataset(tokenizer, text_sequences, max_length):
#     """
#     Encode sequences with Bert-style tokenizer for inputting
#     to the Bert layer.
#     """
#     token_ids = np.zeros(shape=(len(text_sequences), max_length),
#                         dtype=np.int32)

#     for i, text_sequence in enumerate(text_sequences):
#         encoded = tokenizer.encode(text_sequence)
#         token_ids[i, 0:len(encoded)] = encoded
#     attention_masks = (token_ids != 0).astype(np.int32)
#     return {"input_ids": token_ids, "attention_masks": attention_masks}

# def encode_token_labels(text_sequences, tag_names, tokenizer, tag_map,
#                         max_length):
#     encoded = np.zeros(shape=(len(text_sequences), max_length), dtype=np.int32)
#     for i, (text_sequence, word_labels) in enumerate(
#             zip(text_sequences, tag_names)):
#         encoded_labels = []
#         for word, word_label in zip(text_sequence.split(), word_labels.split()):
#             tokens = tokenizer.tokenize(word)
#             encoded_labels.append(tag_map[word_label])
#             expand_label = word_label.replace("B-", "I-")
#             if not expand_label in tag_map:
#                 expand_label = word_label
#             encoded_labels.extend([tag_map[expand_label]] * (len(tokens) - 1))
#         encoded[i, 1:len(encoded_labels) + 1] = encoded_labels
#     return encoded

class DatasetLoader():
    def __init__(self, dset_name):
        self.name = dset_name
        self.dset_dir = os.path.join(ROOT_DIR, dset_name)

    def load_prepare_dataset(self):
        print('Loading data...')
        df_train = pd.read_csv(os.path.join(self.dset_dir, 'train.csv'))
        try:
            df_valid = pd.read_csv(os.path.join(self.dset_dir, 'valid.csv'))
        except FileNotFoundError:
            df_valid = None

        intent2id, id2intent = self.load_intents_map()
        tag2id, id2tag = self.load_tags_map()
        
        return df_train, df_valid, intent2id, id2intent, tag2id, id2tag

    def load_intents_map(self):
        intent_names = Path(os.path.join(self.dset_dir, "vocab.intent")).read_text().split()
        intent2id = dict((label, idx) for idx, label in enumerate(intent_names))
        id2intent = {intent2id[i] : i for i in intent2id.keys()}
        return intent2id, id2intent

    def load_tags_map(self):
        tag_names = Path(os.path.join(self.dset_dir, "vocab.tag")).read_text().strip().splitlines()
        tag_map = {}
        tag2id = dict((label, idx) for idx, label in enumerate(tag_names))
        id2tag = {tag2id[i] : i for i in tag2id.keys()}
        return tag2id, id2tag


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

# END OF TOOLS FOR STANDART DATASETS