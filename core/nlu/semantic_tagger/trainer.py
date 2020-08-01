import pandas as pd
import json
from pathlib import Path
import numpy as np
import argparse
import os
import datetime
from config import NLU_DIR, ROOT_DIR, NLU_CONFIG

from ...tools.utils import TagsDatasetLoader as DatasetLoader
from ...tools.utils import encode_token_labels, encode_dataset, dump
from ...featurizers.transformer_featurizer import SentenceFeaturizer
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from .dense_model import DenseTagsExtractor


parser = argparse.ArgumentParser(description='Trainer for intent classificator')
# solve problem with max length
parser.add_argument('--dataset', type=str, default=NLU_CONFIG['universal_tagger_dataset'], help='Choose dataset dir for training')
parser.add_argument("--rebuild_dataset", default=False, action="store_true")

args = parser.parse_args()


class ModelTrainer():

    def __init__(self, rebuild_dataset, dset_name):

        self.model_save_dir = os.path.join(ROOT_DIR, NLU_CONFIG['sem_tagger_model'])
        self.validate = True

        learning_rate, epsilon = 0.001, 1e-09
        self.params = {'learning_rate' : learning_rate, 'epsilon' : epsilon,
                        'dataset_name' : dset_name}
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

        self.featurizer = SentenceFeaturizer()
        if rebuild_dataset:
            from ...tools.yaml2dataset import DatasetBuilder
            dset = os.path.join(dset_name, 'dataset.yaml')
            db = DatasetBuilder(dset, -1, dset_name)
            db.build_dataset()

        self.dataset_preload(dset_name)
        self.model = DenseTagsExtractor(len(self.id2tag))

        opt = Adam(learning_rate=0.001, epsilon=1e-09)
        losses = [SparseCategoricalCrossentropy(from_logits=True)]

        metrics = [SparseCategoricalAccuracy('accuracy')]
        self.model.compile(optimizer=opt, loss=losses, metrics=metrics)

    def encode_tokenize(self, seq):
        encoded = encode_dataset(self.featurizer, seq).tolist()
        encoded = self.featurizer.encode(
            encoded, output_value='token_embeddings', convert_to_numpy=True, is_pretokenized=True
        )
        return np.array(encoded)


    def dataset_preload(self, dset_name):
        d = DatasetLoader(dset_name)
        df_train, df_valid, intent2id, id2intent, self.tag2id, self.id2tag = d.load_prepare_dataset()
        if df_valid is None:
            self.validate = False
        self.encoded_seq_train = self.encode_tokenize(df_train['words'])
        self.tag_train = encode_token_labels(df_train["words"], df_train["word_labels"], self.featurizer, self.tag2id)

        if self.validate:
            self.encoded_seq_valid = self.encode_tokenize(df_valid['words'])
            self.tag_valid = encode_token_labels(
                df_valid["words"], df_valid["word_labels"], self.featurizer, self.tag2id)

        self.params.update({'output_length' : len(self.tag2id)})


    def train(self, epochs, batch_size):

        time = datetime.datetime.now()
        name = f"tag_extr_e{epochs}_bs{batch_size}"

        checkpoint_path = os.path.join(self.model_save_dir, name)
        # print(self.encoded_seq_train.shape, self.tag_train.shape); sys.exit()

        # Create a callback that saves the model's weights
        cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                        save_weights_only=True,
                                        verbose=1)

        if self.validate:
            history = self.model.fit(self.encoded_seq_train,
                        self.tag_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(self.encoded_seq_valid, self.tag_valid),
                        callbacks=cp_callback)
        else:
            history = self.model.fit(self.encoded_seq_train,
                        self.tag_train, epochs=epochs, batch_size=batch_size,
                        callbacks=cp_callback)
        
        dump(self.params, os.path.join(self.model_save_dir, 'params.json'))



if __name__ == "__main__":
    trainer = ModelTrainer(rebuild_dataset=args.rebuild_dataset, dset_name=args.dataset)
    trainer.train(epochs=9, batch_size=32)
    model = trainer.model