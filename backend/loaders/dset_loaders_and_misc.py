import pandas as pd
import numpy as np
import os
import re
from pathlib import Path
from backend.configuration.config import ROOT_DIR, NLU_CONFIG


# TODO: вынести некоторые функции в root_dir config.py


def space_punct(text):
    """
    Hey,is it question? -> Hey , is it question ?
    """
    return re.sub('(?<! )(?=[.,!?()])|(?<=[.,!?()])(?! )', r' ', text)


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
        intent_names = Path(os.path.join(self.dset_intents_dir, "locale.intent")).read_text().split()
        intent2id = dict((label, idx) for idx, label in enumerate(intent_names))
        id2intent = {intent2id[i] : i for i in intent2id.keys()}
        return intent2id, id2intent

    def load_tags_map(self, dset_name=None):
        if dset_name is not None:
            self.dset_tags_dir = os.path.join(ROOT_DIR, dset_name)
        tag_names = Path(os.path.join(self.dset_tags_dir, "locale.tag")).read_text().strip().splitlines()
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

        df = pd.read_csv(os.path.join(self.dset_intents_dir, 'train.csv'))
        df_valid = None
        if 'valid.csv' in os.listdir(self.dset_intents_dir):
            df_valid = pd.read_csv(os.path.join(self.dset_intents_dir, 'valid.csv'))
        return df, df_valid, intent2id, id2intent, None, None

class TagsDatasetLoader(DatasetLoader):
    def __init__(self, dset_intents_name, dset_tags_name=None):
        super().__init__(dset_intents_name, dset_tags_name=dset_tags_name)

    def load_prepare_dataset(self):
        print('Loading data...')

        tag2id, id2tag = self.load_tags_map()
        intent2id, id2intent = self.load_intents_map()

        df = pd.read_csv(os.path.join(self.dset_tags_dir, 'train.csv'))
        df_valid = None
        if 'valid.csv' in os.listdir(self.dset_tags_dir):
            df_valid = pd.read_csv(os.path.join(self.dset_tags_dir, 'valid.csv'))
        return df, df_valid, intent2id, id2intent, tag2id, id2tag

def load_intents_map(intents_dset_name):
    d = DatasetLoader(dset_intents_name=intents_dset_name)
    return d.load_intents_map()

def load_tags_map(tags_dset_name):
    d = DatasetLoader(dset_intents_name=tags_dset_name)
    return d.load_tags_map()

def load_id2intent(dset_name):
    if dset_name is not None:
        dset_intents_dir = os.path.join(ROOT_DIR, dset_name)
    intent_names = Path(os.path.join(dset_intents_dir, "locale.intent")).read_text().split()
    id2intent = dict((idx, label) for idx, label in enumerate(intent_names))
    return id2intent

def load_id2tag(dset_name):
    if dset_name is not None:
        dset_tags_dir = os.path.join(ROOT_DIR, dset_name)
    tag_names = Path(os.path.join(dset_tags_dir, "locale.tag")).read_text().strip().splitlines()
    id2tag = dict((idx, label) for idx, label in enumerate(tag_names))
    return id2tag

def load_map(self, text_vocab_path):
    """
    Path to the locale.__entity_name__
    This file located at dataset dir
    """
    if dset_name is not None:
        path = os.path.join(ROOT_DIR, text_vocab_path)
    names = Path(path).read_text().strip().splitlines()
    ent_map = {}
    ent2id = dict((label, idx) for idx, label in enumerate(names))
    id2ent = {ent2id[i] : i for i in ent2id.keys()}
    return ent2id, id2ent

class DatasetsTools():
    def __init__(self):
        pass
    
    def csv_line2yaml_line(self, a, b):
        """
        Args:
            a: str, 'Add Don and Sherri to my Meditate to Sounds of Nature playlist'
            b: str, 'O B-entity_name I-entity_name I-entity_name O B-playlist_owner B-playlist I-playlist I-playlist I-playlist I-playlist O'
        Out:
            str, 'Add [entity_name](Don and Sherri) to [playlist_owner](my) [playlist](Meditate to Sounds of Nature) playlist'
        """
        a = a.split()
        b = b.split()
        sen = []
        tag_name = ''
        tag_seq = {}
        for i in range(len(b)):
            tag = b[i]
            word = a[i]
            if tag == 'O':
                if len(tag_seq) != 0:
                    sen.append(f"[{tag_name}]" + '(' + ' '.join(tag_seq[tag_name]) + ')')
                    tag_seq = {}
                sen.append(word)
            else:
                tag_name_now = tag.replace('B-', '').replace('I-', '')
                if 'B-' in tag:
                    if len(tag_seq) != 0 and tag_name != '':
                        sen.append(f"[{tag_name}]" + '(' + ' '.join(tag_seq[tag_name]) + ')')
                        tag_seq == {}
                    tag_seq.update({tag_name_now : [word]})
                    tag_name = tag_name_now
                elif 'I-' in tag:
        #             print(tag_seq, sen)
                    tag_seq[tag_name_now].append(word)

        if len(tag_seq) != 0:
            sen.append(f"[{tag_name}]" + '(' + ' '.join(tag_seq[tag_name]) + ')')
        sen = ' '.join(sen)
        return sen
    
    def get_from_df_dataset(self, df):
        intents = {}
        tags = {}
        err_cnt = 0
        for (i, r) in df.iterrows():
            intent_name = r['intent_label']
            try:
                line = self.csv_line2yaml_line(r['words'], r['word_labels'])
                line_tags = self.read_tags([line])[0]
                if intent_name in intents.keys():
                    intents[intent_name].append(line)
                    tags[intent_name].append(line_tags)
                else:
                    intents.update({intent_name : [line]})
                    tags.update({intent_name : [line_tags]})
            except IndexError:
                err_cnt += 1
                pass
        return intents, tags

    def read_tags(self, raw_intents):
        """
        Extract tags from yaml line
        """
        intents = []
        for i in raw_intents:
            tag_class_ids = [j.span() for j in re.finditer(r'\[(.*?)\]', i)]
            tag_class, tag_names = [], []
            for t in tag_class_ids:
                tag_class.append(i[t[0]:t[1]].replace('[', '').replace(']', ''))
            i = re.sub(r'\[(.*?)\]', '', i)
            tag_ids = [j.span() for j in re.finditer(r'\((.*?)\)', i)]
            for t in tag_ids:
                tag_names.append(i[t[0]:t[1]].replace('(', '').replace(')', ''))
            tag_map = list(zip(tag_class, tag_names))
            intents.append(tag_map)
        return intents
    
    def get_all_intent_names(self, dset_list):
        names = []
        for i in dset_list:
            names.append(i['name'])
        return list(set(names))
    
    def find_by_intent(self, dset_list):
        pass
    
    def append_intent2yaml(self, extracted_intents, yaml_dset, task='train'):
        for intent in extracted_intents.keys():
            if not intent in get_all_intent_names(yaml_dset['intents']):
                intent_frame = {'type' : 'intent', 'task' : task, \
                               'name' : intent, 'utterances' : extracted_intents[intent]}
                yaml_dset['intents'].append(intent_frame)
            else:
                pass
        return yaml_dset
    
    def append_tag2yaml(self, extracted_entities, yaml_dset):
        tag_by_intents = {}
        for intent in tags.keys():
            tag_by_intents.update({intent : {} })
            tag_list = list(itertools.chain.from_iterable(tags[intent]))
            for pair in tag_list:
                tag_name, tag = pair
                if tag_name in tag_by_intents[intent].keys():
                    tag_by_intents[intent][tag_name].append(tag)
                else:
                    tag_by_intents[intent].update({tag_name : [tag]})
            for tag_name in tag_by_intents[intent].keys():
                tag_by_intents[intent][tag_name] = list(set(tag_by_intents[intent][tag_name]))
        tag_by_intents
    
    def dataset2yaml(self, from_path, to_path='data/nlu_data/standard/dataset.yaml'):
        from_path = os.path.join(ROOT_DIR, from_path)
        yaml_path = os.path.join(ROOT_DIR, to_path)
        file = open(yaml_path)
        dset = yaml.load(file, Loader=yaml.FullLoader)
        train_path = os.path.join(from_path, 'train.csv')
        intents_train, tags_train = self.get_from_df_dataset(pd.read_csv(train_path))
        dset = self.append_intent2yaml(intents_train, dset, 'train')
        if 'valid.csv' in os.listdir(from_path):
            intents_valid, tags_valid = self.get_from_df_dataset(pd.read_csv(os.path.join(from_path, 'valid.csv')))
            dset = self.append_intent2yaml(intents_train, dset, 'valid')
        file = open(yaml_path, 'w')
        dset = yaml.dump(dset,file, sort_keys=True)


# END OF TOOLS FOR STANDART DATASETS

def get_all_intents(dataset):
    intents_vocab_list = []
    dataset = os.path.join(ROOT_DIR, dataset, 'locale.intent')
    with open(dataset, 'r') as file:
        for i in file.readlines():
            intents_vocab_list.append(i.replace('\n', ''))
    return intents_vocab_list