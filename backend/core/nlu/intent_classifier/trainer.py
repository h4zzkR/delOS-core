import pandas as pd
import json
from pathlib import Path
import argparse
import os
import datetime
from config import NLU_DIR, ROOT_DIR, NLU_CONFIG

from ..tools.utils import IntentsDatasetLoader as DatasetLoader #, encode_dataset, encode_token_labels
from ..tools.utils import dump
from ..featurizers.transformer_featurizer import SentenceFeaturizer
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.client import device_lib
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from .dense_model import DenseClassifierModel as Classifier
from ..tools.yaml2dataset import DatasetBuilder


parser = argparse.ArgumentParser(description='Trainer for intent classificator')
# solve problem with max length
parser.add_argument('--model', type=str, default='dense', help='Model type (logreg, dense, etc)')
parser.add_argument('--dataset', type=str, default=NLU_CONFIG['intents_classificier_dataset'], help='Choose dataset dir for training')
parser.add_argument("--rebuild_dataset", default=False, action="store_true")

args = parser.parse_args()


class ModelTrainer():

    def __init__(self, model_name, rebuild_dataset, dset_name):
        self.model_name = model_name
        self.validate = True

        learning_rate, epsilon = 0.001, 1e-09
        self.params = {'learning_rate' : learning_rate, 'epsilon' : epsilon,
                        'dataset_name' : dset_name}

        self.model_save_dir = os.path.join(NLU_DIR, 'intent_classifier/model')
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

        self.featurizer = SentenceFeaturizer()
        if rebuild_dataset:
            dset = os.path.join(dset_name, 'dataset.yaml')
            db = DatasetBuilder(dset, -1, dset_name)
            db.build_dataset()

        self.dataset_preload(dset_name)

        self.model = Classifier(
                    intent_num_labels=len(self.intent2id),
                    )
                    

        opt = Adam(learning_rate=learning_rate, epsilon=epsilon)
        losses = [SparseCategoricalCrossentropy(from_logits=False)]

        metrics = [SparseCategoricalAccuracy('accuracy')]
        self.model.compile(optimizer=opt, loss=losses, metrics=metrics)

    def dataset_preload(self, dset_name):
        d = DatasetLoader(dset_name)
        df_train, self.y_train, df_valid, self.y_valid, \
            self.intent2id, self.id2intent, tag2id, id2tag = d.load_prepare_dataset()
        self.encoded_train = self.featurizer.encode(df_train['words'])

        if self.y_valid is None:
            self.validate = False

        if self.validate:
            self.encoded_valid = self.featurizer.encode(df_valid['words'])

        self.params.update({'output_length' : len(self.intent2id)})


    def train(self, epochs, batch_size):

        time = datetime.datetime.now()
        name = f"intents_{self.model_name}_cls_e{epochs}_bs{batch_size}"

        checkpoint_path = os.path.join(self.model_save_dir, name)

        # Create a callback that saves the model's weights
        cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                        save_weights_only=True,
                                        verbose=1)

        if self.validate:
            history = self.model.fit(self.encoded_train,
                        self.y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(self.encoded_valid, self.y_valid),
                        callbacks=cp_callback)
        else:
            history = self.model.fit(self.encoded_train,
                        self.y_train, epochs=epochs, batch_size=batch_size,
                        callbacks=cp_callback)

        dump(self.params, os.path.join(self.model_save_dir, 'params.json'))




if __name__ == "__main__":
    trainer = ModelTrainer(model_name=args.model, \
        rebuild_dataset=args.rebuild_dataset, dset_name=args.dataset)
    trainer.train(epochs=9, batch_size=32)
    model = trainer.model