import os
import yaml
import tqdm
import argparse
import re
import pandas as pd
from pathlib import Path
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from backend.configuration.config import INTENTS, ENTITIES, UTTERANCES, CUSTOM_NERINTENT_DSET
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as naf
from nlpaug.util import Action

# Translate dataset from yaml type
# to pandas dataframe with mapping


class IntentAug:
    # """
    # Агументация intent-ов
    # Работает не всегда предсказуемо, поэтому для аугментации тегов
    # нужно применять что-то более умное
    # """
    def __init__(self, num_of_samples=2, swap=True):
        self.num_augs_on_sentence = num_of_samples
        self.swap_words = swap
        self.big_augs = naf.Sequential([
            # naw.SynonymAug(aug_src='wordnet'),
            # naw.BackTranslationAug(
            #     from_model_name='transformer.wmt19.en-de',
            #     to_model_name='transformer.wmt19.de-en'
            # )
            naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="insert")
        ])
        self.small_augs = naf.Sequential([naw.SynonymAug(aug_src='wordnet')])
        self.swap_aug = naw.RandomWordAug(action="swap")

    def augment(self, example):
        if (len(example.split()) <= 4):
            augs = [self.small_augs.augment(example) for _ in range(self.num_augs_on_sentence)]
        else:
            augs = [self.big_augs.augment(example) for _ in range(self.num_augs_on_sentence)]
            if self.swap_words:
                augs_ = list(augs)
                for i in augs_:
                    for _ in range(self.num_augs_on_sentence):
                        swapped = self.swap_aug.augment(i)
                        augs.append(swapped)
        return list(set(augs))


class DatasetTranslator:
    def __init__(self, path2dset=CUSTOM_NERINTENT_DSET, max_synonyms=2, path2write=None,
                 validate_split=True, augment_intents=True, drop_stopwords=False):
        """
        Парсит yaml датасет и сохраняет/возвращает pandas dataframe, готовый для обучения
        path2dset: FULL path to the dataset (default is custom)
        max_synonyms: check yaml custom dataset for more info
        path2write: dump translated dataset to this path (full path)
        """
        self.path2dset = path2dset
        self.path = path2dset
        self.out = path2write
        self.max_synonyms = max_synonyms
        self.intent_vocab = []
        self.tag_vocab = []

        self.validate_split = validate_split
        self.augment_intents = augment_intents
        self.intent_augmentor = IntentAug(swap=True)
        if augment_intents:
            print('intents will be augmented')

    def write_intent_vocab(self):
        with open(os.path.join(self.out, 'vocab.intent'), 'w') as file:
            for i in self.intent_vocab:
                file.write(i + '\n')

    def write_tag_vocab(self):
        with open(os.path.join(self.out, 'vocab.tag'), 'w') as file:
            for i in self.tag_vocab:
                file.write('B-' + i + '\n')
                file.write('I-' + i + '\n')
            file.write('O' + '\n')

    def build_dataset(self):
        """
        Translates yaml and returns pandas dataframe with it
        or just dumps it to path
        """
        dset = self.read_yaml()
        raw_intents, raw_entities = dset[INTENTS], dset[ENTITIES]
        self.entities_vocab = self.build_entities_vocab(raw_entities)
        print('building maps for intent tempaltes...')
        processed_intents = self.read_slots_from_yaml(raw_intents)
        print('starting to combine entities...')
        df = self.build_combination_sequence(processed_intents)
        length = len(df)
        print(f'{length} unique examples builded')

        # Shuffle
        df = shuffle(df).reset_index(drop=True)

        if self.validate_split:
            df, valid = train_test_split(df, test_size=0.15, random_state=42)
        if self.out:
            tp = os.path.join(self.out, 'train.csv')
            vp = os.path.join(self.out, 'valid.csv')
            if self.validate_split:
                df.to_csv(tp, index=False)
                valid.to_csv(vp, index=False)
            else:
                df.to_csv(tp, index=False)
            self.write_intent_vocab();
            self.write_tag_vocab()
        else:
            if self.validate_split:
                return df, valid
            return df, (self.intent_vocab, self.tag_vocab)

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
            vocab.update({name: values})
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

    def make_line(self, iclass, intent, imap, aug=False):
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
                    mask[pos + 1] = 'I-' + slot_class
            else:
                mask[pos] = 'B-' + slot_class
        text = intent.replace('(', '').replace(')', '')
        mask = ' '.join(mask)
        return (iclass, text, mask, intent_len, aug)

    def build_combination_sequence(self, p_intents):
        df = pd.DataFrame(columns=['intent_label', 'words', 'word_labels', 'length', 'aug'])
        cnter = 0
        for i in p_intents:
            intent_class, intent, map_ = i
            data_line = self.make_line(intent_class, intent, map_, aug=False)
            if (self.augment_intents):
                # print('starting intents augmentation')
                augmented = self.intent_augmentor.augment(data_line[1])
                for j in range(len(augmented)):
                    df.loc[cnter + j] = (data_line[0], augmented[j], data_line[2], len(augmented[j].split()), True)
                cnter += len(augmented)
            df.loc[cnter] = data_line
            cnter += 1
        return df

# if __name__ == "__main__":
#     builder = DatasetBuilder(path2dset=args.path2dset,
#                             max_synonyms=args.max_synonyms,
#                             path2write=args.path2write)
#     builder.build_dataset()