import itertools
import os
import re
from collections import defaultdict

import nlpaug.augmenter.word as naw
import nlpaug.flow as naf
import pandas as pd

from backend.configuration.config import SKILLS_PATH
from backend.functional import listdir, read_yaml


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


class SkillsTranslator:
    def __init__(self, path=SKILLS_PATH, tagger=True, validate_split=True, augment_intents=True,
                 dump_path=os.path.join(SKILLS_PATH, 'dumped_dataset.csv'),
                 locale="en-us"):
        """
        Parse skills from skill folder path from yaml to train-ready format
        tagger: also generate tag data from entities for semantic taggers train
        validate_split: leave a part for validate
        augment_intents: augment intents with nlpaug
        dump_path: dump translated yamls to drive
        """
        self.path = path
        self.val_split = validate_split
        self.augment_intents = augment_intents
        self.dump = dump_path
        self.locale = locale

        self.df = pd.DataFrame(columns=['skill_label', 'intent_label', 'words', 'word_labels', 'length', 'aug'])
        self.cnter = 0

        if augment_intents:
            print('intents will be augmented')

    def read_skill(self, path):
        ls = listdir(path)

    def read_skills_dir(self):
        """
        Parse contents of path, return map {locale}{*locale*}{intent_name} : [files]
        """
        walker = os.walk(self.path)
        _, skills_list, _ = [i for i in next(walker)]
        skills_list[:] = [d for d in skills_list if not d[0] == '.']
        skills_files = {'locale': defaultdict(lambda: defaultdict(list))}
        dialogs_files = {'locale': defaultdict(lambda: defaultdict(list))}
        for root, dirs, files in os.walk(self.path):
            files = [f for f in files if not f[0] == '.']
            if len(files) == 0:
                continue
            dirs[:] = [d for d in dirs if not d[0] == '.']
            locale = root[root.rfind('/') + 1:]
            if locale != self.locale:
                continue
            skill_name = root[root.rfind('skills') + len('skills') + 1:]
            skill_name = skill_name[:skill_name.find('/')]
            for f in files:
                if 'intent' in f:
                    skills_files['locale'][self.locale][skill_name].append(os.path.join(root, f))
                elif 'dialog' in f:
                    dialogs_files['locale'][self.locale][skill_name].append(os.path.join(root, f))
        return (dialogs_files, skills_files)

    def expand_utterences(self, skill_label, intent_label, utterences):
        expanded = []
        for ut in utterences:
            keywords = {i: i[1:-1].split(' | ') for i in re.findall('\(.*?\)', ut)}
            # Get a list of bracketed terms
            lsources = [i for i in keywords.keys()]
            # Build a list of the possible substitutions
            ldests = []
            for source in lsources:
                ldests.append(keywords[source])

            # Generate the various pairings
            for lproduct in itertools.product(*ldests):
                output = ut
                for src, dest in zip(lsources, lproduct):
                    #         # Replace each term (you could optimise this using a single re.sub)
                    output = output.replace("%s" % src, dest)
                self.df.loc[self.cnter] = (skill_label, intent_label, output, 'None', len(output.split()), False)
                self.cnter += 1
                # expanded.append((output, len(output.split())))
        return expanded

    def read_parse_skill(self, skill_label):
        subs = self.skills_files['locale'][self.locale][skill_label]
        for f in subs:
            skill_yaml = read_yaml(f)
            skill_label = f[f.rfind("/") + 1:].replace('.intent.yaml', '')
            for intent in skill_yaml['intents']:
                intent_label = intent['name']
                utterences = self.expand_utterences(skill_label, intent_label, intent['utterances'])

    def build_dataset(self):
        _, self.skills_files = self.read_skills_dir()
        # df = pd.DataFrame(columns=['skill_label', 'intent_label', 'words', 'word_labels', 'length', 'aug'])
        for skill in self.skills_files['locale'][self.locale].keys():
            self.read_parse_skill(skill)

        if (self.dump is not None):
            self.df.to_csv(self.dump, index=False)
        return self.df



if __name__ == "__main__":
    st = SkillsTranslator()
    st.build_dataset()
#     builder = DatasetBuilder(path2dsets=args.path2dsets,
#                             max_synonyms=args.max_synonyms,
#                             path2write=args.path2write)
#     builder.build_dataset()
