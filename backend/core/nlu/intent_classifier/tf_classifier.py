import os
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from backend.functional import elapsed_time
from tensorflow.keras.optimizers import Adam
from backend.backbones.nlu_unit import ModuleUnit
from backend.functional import tf_set_memory_growth
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from backend.configuration.config import ROOT_DIR, CLASSIFIER_DIR, NLU_CONFIG, MODELS_FIT_PARAMS
from backend.core.nlu.intent_classifier.dense_model import DenseClassifierModel
from backend.core.nlu.tools.utils import IntentsDatasetLoader as DatasetLoader
from backend.functional import dump, jsonread
from backend.core.nlu.featurizers.transformer_featurizer import SentenceFeaturizer

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.flow as naf
from nlpaug.util import Action

class IntentAug:
    def __init__(self, num_of_samples=3, swap=True):
        self.num_of_samples = 3
        self.swap = swap
        self.build_augments()
    
    def build_augments(self):
        self.augs = naf.Sequential([
            naw.SynonymAug(aug_src='wordnet'),
        #     naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="insert"),
        #     naw.RandomWordAug(action="swap")
        ])
        
    def augment(self, example):
        augs = [self.augs.augment(example) for _ in range(self.num_of_samples)]
        if self.swap:
            swap_aug = naw.RandomWordAug(action="swap")
            augs_ = list(augs)
            for i in augs_:
                for _ in range(self.num_of_samples):
                    swapped = swap_aug.augment(i)
                    augs.append(swapped)
        return augs

class IntentClassifier(ModuleUnit):
    def __init__(self, model_name='intent_classifier'):
        super().__init__(model_name)
        self.base_model_ = 'intent_classifier_base_model'
        self.model_dir = os.path.join(CLASSIFIER_DIR, 'model')
        self.fitted = False

    def _load_base_model(self, out_dim):
        if str(NLU_CONFIG[self.base_model_]) == 'dense':
            if self.fit_params['from_logits']:
                model = DenseClassifierModel(out_dim, from_logits=True)
            else:
                model = DenseClassifierModel(out_dim, from_logits=False)
        else:
            return IndentationError
        return model

    def _load_fit_parameters(self):
        self.fit_params = jsonread(MODELS_FIT_PARAMS)[self.base_model_]
        self.fit_params['validate'] = True
        if self.fit_params['validate_prob'] is None or self.fit_params['validate_prob'] == 1:
            self.fit_params['validate'] = False

    def load(self):
        checkpoint_path = os.path.join(ROOT_DIR, NLU_CONFIG['classifier_model'])
        try:
            ckp = Path(os.path.join(checkpoint_path, 'checkpoint')).read_text()
            m = ckp.splitlines()[0].split(': ')[-1]
            self.fit_params = jsonread(os.path.join(checkpoint_path , 'params.json'))
            checkpoint_path = os.path.join(checkpoint_path, m[1:-1])
            # _, self.id2tag = load_map(os.path.join(self.fit_params["dataset_name"], 'locale.tag'))

            self.model = self._load_base_model(self.fit_params["output_length"])
            self.model.load_weights(checkpoint_path)
            self.fitted = True
        except FileNotFoundError:
            print(f'Classifier model not fitted, needs training')
            self.fitted = False

    def _load_format_dataset(self, dataset, augment):
        d = DatasetLoader(dataset)
        df_train, df_valid, self.intent2id, self.id2intent, tag2id, id2tag = d.load_prepare_dataset()

        if augment:
            # LOGGING
            print(f"Data augmentation enabled")
            length = df_train.shape[0]
            augmented_df = []
            augmentor = IntentAug()
            for (_, r) in tqdm(df_train.iterrows()):
                intent_name = r['intent_label']
                augmented = list(map(lambda seq: [intent_name, seq, None, None], augmentor.augment(r['words'])))
                augmented_df += augmented
            augmented_df = pd.DataFrame(augmented_df, columns=['intent_label', 'words', 'word_labels', 'length'])
            df_train = pd.concat([df_train, augmented_df])
            # LOGGING
            print(f"Dataset augmented from {length} examples to {df_train.shape[0]} examples")

        if df_valid is not None:
            self.fit_params['validate'] = True
        self.fit_params['dataset_name'] = dataset

        featurizer = SentenceFeaturizer()

        encoded_valid, y_valid = None, None
        if not self.fit_params['validate']:
            validate_prob = self.fit_params['validate_prob']
            length = len(df_train)
            df_valid, df_train = df_train.loc[:int(validate_prob * length)], df_train.loc[int(validate_prob * length):]
            self.fit_params['validate'] = True

        encoded_valid = featurizer.encode(df_valid['words'])
        y_valid = df_valid['intent_label'].map(self.intent2id).values

        encoded_train = featurizer.encode(df_train['words'])
        y_train = df_train['intent_label'].map(self.intent2id).values

        self.fit_params['output_length'] = len(self.intent2id)
        return (encoded_train, y_train), (encoded_valid, y_valid)


    def fit(self, dataset, augment=True):
        """
        Dataset name, intent class
        """
        self._load_fit_parameters()
        (x_train, y_train), (x_valid, y_valid) = self._load_format_dataset(dataset, augment)
        self.model = self._load_base_model(self.fit_params['output_length'])

        opt = Adam(learning_rate=self.fit_params['learning_rate'], epsilon=self.fit_params['epsilon'])
        if self.fit_params['from_logits']:
            loss = SparseCategoricalCrossentropy(from_logits=True)
        else:
            loss = SparseCategoricalCrossentropy(from_logits=False)
        metrics = SparseCategoricalAccuracy('accuracy')
        self.model.compile(optimizer=opt, loss=loss, metrics=metrics)

        time = datetime.datetime.now()

        checkpoint_path = os.path.join(self.model_dir, f"classifier")
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        cp_callback = ModelCheckpoint(filepath=os.path.join(checkpoint_path, f"classifier_model"), \
                                        save_weights_only=True, verbose=1, save_best_only=True)
        if self.fit_params['validate']:
            history = self.model.fit(x_train, y_train,
                        epochs=self.fit_params['epochs'], batch_size=self.fit_params['batch_size'],
                        validation_data=(x_valid, y_valid), callbacks=cp_callback)
        else:
            history = self.model.fit(x_train, y_train,
                        epochs=self.fit_params['epochs'], batch_size=self.fit_params['batch_size'],
                        callbacks=cp_callback)
        
        dump(self.fit_params, os.path.join(checkpoint_path, 'params.json'))
        self.fitted = True

        # LOGGING
        print(f"Classifier model fitted in {elapsed_time(time)}s")

    def classify(self, inputs):
        out = self.model.predict(inputs)
        out = np.squeeze(np.array(out))
        class_id = np.argmax(out)
        return class_id, out
        
    def test_classifier(self, dataset='data/nlu_data/standard'):
        d = DatasetLoader(dataset)
        df_train, df_valid, intent2id, id2intent, tag2id, id2tag = d.load_prepare_dataset()
        self.load()
        ft = SentenceFeaturizer()
        while True:
            inp_str = input()
            inp = ft.encode(inp_str)
            in_id, out = self.classify(inp)
            print(id2intent[in_id])

if __name__ == "__main__":
    tf_set_memory_growth()
    obj = IntentClassifier()
    obj.fit('data/nlu_data/standard')
    # d = DatasetLoader('data/nlu_data/standard')
    # df_train, df_valid, intent2id, id2intent, tag2id, id2tag = d.load_prepare_dataset()
    # obj.load()
    # ft = SentenceFeaturizer()
    # while True:
    #     inp_str = input()
    #     inp = ft.encode(inp_str)
    #     in_id, out = obj.classify(inp)
    #     print(id2intent[in_id])