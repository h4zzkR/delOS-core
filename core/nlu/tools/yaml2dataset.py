import os
import yaml
import tqdm
import argparse
import re
import pandas as pd
from pathlib import Path
from config import ROOT_DIR, INTENTS, ENTITIES, UTTERANCES

parser = argparse.ArgumentParser(
    description='Tool that combines entities and intents for building datastet from human-friendly yaml'
    )
parser.add_argument('--path2dset', type=str, default='data/nlu_data/custom/dataset.yaml')
parser.add_argument('--path2write', type=str, default='data/nlu_data/custom/train.csv')
parser.add_argument('--max_synonyms', type=int, default=-1, help='Set the maximum of synonym numbers for one intent template')

args = parser.parse_args()
#TODO: убрать комбинации шаблонных, аугментации, мб на основе рандома решать, какие слоты заменять синонимами

class DatasetBuilder:
    def __init__(self, path2dset, max_synonyms, path2write=None, drop_stopwords=False):
        # TODO: add synonym augmentions
        # TODO: classes normalization: too many classes of ones
        self.path = Path(os.path.join(ROOT_DIR, path2dset))
        self.out = os.path.join(ROOT_DIR, path2write)
        self.max_synonyms = max_synonyms
        self.intent_vocab = []
        self.tag_vocab = []

    def write_intent_vocab(self):
        with open(os.path.join(self.path.parent, 'vocab.intent'), 'w') as file:
            for i in self.intent_vocab:
                file.write(i + '\n')

    def write_tag_vocab(self):
        with open(os.path.join(self.path.parent, 'vocab.tag'), 'w') as file:
            for i in self.tag_vocab:
                file.write('B-' + i + '\n')
                file.write('I-' + i + '\n')

    def augment_intent(slot_map, intent):
        pass

        
    def build_dataset(self):
        dset = self.read_yaml()
        raw_intents, raw_entities = dset[INTENTS], dset[ENTITIES]
        self.entities_vocab = self.build_entities_vocab(raw_entities)
        print('building maps for intent tempaltes...')
        processed_intents = self.read_slots_from_yaml(raw_intents)
        print('starting to combine entities...')
        df = self.build_combination_sequence(processed_intents)
        print(f'{len(df)} unique examples builded')
        self.write_intent_vocab(); self.write_tag_vocab()
        if self.out:
            df.to_csv(self.out, index=False)
        else:
            return df
        
    def read_yaml(self):
        with open(self.path) as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
        return data
        
    def build_entities_vocab(self, raw_entities):
        vocab = {}
        for e in raw_entities:
            values = []
            for v in e['values']:
                values.append(v[:self.max_synonyms])
            name = f"{e['type']}:{e['name']}"
            vocab.update({name : values})
            self.tag_vocab.append(name)
        return vocab
    
    def read_slots_from_yaml(self, raw_intents):
        intents = []
        for intent_class in raw_intents:
            intent_name = intent_class['name']
            self.intent_vocab.append(intent_name)
            for i in intent_class[UTTERANCES]:
                tag_class_ids = [j.span() for j in re.finditer(r'\[(.*?)\]', i)]
                tag_class, tag_names = [], []
                for t in tag_class_ids:
                    tag_class.append(i[t[0]:t[1]].replace('[', '').replace(']', ''))
                i = re.sub(r'\[(.*?)\]', '', i)
                tag_ids = [j.span() for j in re.finditer(r'\((.*?)\)', i)]
                for t in tag_ids:
                    tag_names.append(i[t[0]:t[1]].replace('(', '').replace(')', ''))
                tag_map = list(zip(tag_class, tag_names))
                intent_pair = [intent_name, i, tag_map]
                intents.append(intent_pair)
        return intents
    
    def make_line(self, iclass, intent, imap):
        intent_len = len(intent.split())
        mask = ['O'] * intent_len
        for key in imap:
            slot_class, slot = key
            # pos =
            pos = [j.span() for j in re.finditer(f'({slot})', intent)][0]
            pos = len(intent[:pos[0]].split()) - 1
            slot_len = len(slot.split())
            if slot_len > 1:
                mask[pos] = 'B-' + slot_class
                for i in range(1, slot_len):
                    mask[pos+1] = 'I-' + slot_class
            else:
                mask[pos] = 'B-' + slot_class
        text = intent.replace('(', '').replace(')', '')
        mask = ' '.join(mask)
        return (iclass, text, mask, intent_len)
                              
            
    def build_combination_sequence(self, p_intents):
        df = pd.DataFrame(columns=['intent_label', 'words', 'word_labels', 'length'])
        cnter = 0
        for i in p_intents:
            intent_class, intent, map_ = i
            df.loc[cnter] = self.make_line(intent_class, intent, map_)
            cnter += 1

        return df
            

if __name__ == "__main__":
    builder = DatasetBuilder(path2dset=args.path2dset,
                            max_synonyms=args.max_synonyms,
                            path2write=args.path2write)
    builder.build_dataset()