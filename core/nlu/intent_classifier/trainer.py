import pandas as pd
from pathlib import Path
import argparse
import os
import datetime
from config import ROOT_DIR

from ..tools.utils import DatasetLoader, encode_dataset, encode_token_labels
from transformers import BertTokenizer
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.client import device_lib
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from .bert_model import BertClassifierModel as Classifier


parser = argparse.ArgumentParser(description='Trainer for intent classificator')
parser.add_argument('--model-name', type=str, default='bert-base-cased')
# solve problem with max length

parser.add_argument('--max_length', type=int, default=49, help='Max tokens in sequence')
parser.add_argument('--dataset', type=str, default='merged', help='Choose dataset dir for training')


args = parser.parse_args()


class ModelTrainer():

    def __init__(self, max_length, name='intents_classifier', dset_name='merged',
                 bert_model_name="bert-base-cased", model_save_dir='nlu/intent_classifier/model'):

        self.training_model = name
        tokenizer = BertTokenizer.from_pretrained(bert_model_name)

        self.model_save_dir = os.path.join(ROOT_DIR, model_save_dir)
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

        self.dataset_preload(dset_name, tokenizer, max_length)

        self.model = Classifier(
                    intent_num_labels=len(self.intent2id),
                    bert_model=bert_model_name,
                    )
                    

        opt = Adam(learning_rate=3e-5, epsilon=1e-08)
        losses = [SparseCategoricalCrossentropy(from_logits=True)]

        metrics = [SparseCategoricalAccuracy('accuracy')]
        self.model.compile(optimizer=opt, loss=losses, metrics=metrics)


    def dataset_preload(self, dset_name, tokenizer, max_length):
        d = DatasetLoader(dset_name)
        df_train, df_valid, self.intent2id, self.id2intent, tag2id, id2tag = d.load_prepare_dataset()

        self.intent_train = df_train["intent_label"].map(self.intent2id).values
        if df_valid is not None:
            self.intent_valid = df_valid["intent_label"].map(self.intent2id).values
        else:
            self.intent_valid = None

        self.encoded_train = encode_dataset(tokenizer, df_train["words"], max_length)
        if self.intent_valid is not None:
            self.encoded_valid = encode_dataset(tokenizer, df_valid["words"], max_length)
        else:
            self.encoded_valid = None


    def train(self, epochs, batch_size):

        time = datetime.datetime.now()
        name = f"intents_cls_e{epochs}_bs{batch_size}"

        checkpoint_path = os.path.join(self.model_save_dir, name)

        # Create a callback that saves the model's weights
        cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                        save_weights_only=True,
                                        verbose=1)

        if self.encoded_valid:
            history = self.model.fit(self.encoded_train,
                        self.intent_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(self.encoded_valid, self.intent_valid),
                        callbacks=cp_callback)
        else:
            history = self.model.fit(self.encoded_train,
                        self.intent_train, epochs=epochs, batch_size=batch_size,
                        callbacks=cp_callback)




if __name__ == "__main__":
    trainer = ModelTrainer(args.max_length,
            bert_model_name=args.model_name, dset_name=args.dataset)
    trainer.train(epochs=2, batch_size=32)
    model = trainer.model

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    while True:
        inp = input()
        inp = tf.constant(tokenizer.encode(inp))[None, :]
        out = model(inp).numpy().argmax(axis=1)[0]
        class_id = model(inp).numpy().argmax(axis=1)[0]
        # return class_id
        print(trainer.id2intent[class_id])