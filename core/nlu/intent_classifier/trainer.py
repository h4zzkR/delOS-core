# Do not forget to fucking run with -m prefix
# and change '/' to '.'

"""
Intents classificator module trainer
"""
import pandas as pd
from pathlib import Path
import argparse
import os
import datetime
from ..nlu_data.utils import DatasetLoader, encode_dataset, encode_token_labels
from transformers import BertTokenizer
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.client import device_lib
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from .base import RequestIntentClassifier


parser = argparse.ArgumentParser(description='Trainer for bert-based intent classificator')
parser.add_argument('--model-name', type=str, default='bert-base-cased')
# solve problem with max length

parser.add_argument('--max_length', type=int, default=49, help='Max tokens in sequence')
parser.add_argument('--dataset', type=str, default='merged', help='Choose dataset dir for training')


args = parser.parse_args()


class ModelTrainer():

    def __init__(self, max_length, curdir, name='intents_classifier', dset_name='merged',
                 bert_model_name="bert-base-cased", model_save_dir='model'):

        self.training_model = name
        tokenizer = BertTokenizer.from_pretrained(bert_model_name)

        self.model_save_dir = os.path.join(curdir, model_save_dir)
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

        self.dataset_preload(dset_name, curdir, tokenizer, max_length)

        self.model = RequestIntentClassifier(
                    intent_num_labels=len(self.intent2id),
                    bert_model=bert_model_name,
                    )
                    

        opt = Adam(learning_rate=3e-5, epsilon=1e-08)
        losses = [SparseCategoricalCrossentropy(from_logits=True)]

        metrics = [SparseCategoricalAccuracy('accuracy')]
        self.model.compile(optimizer=opt, loss=losses, metrics=metrics)


    def dataset_preload(self, dset_name, curdir, tokenizer, max_length):
        d = DatasetLoader(dset_name)
        df_train, df_valid, df_test, \
                self.intent2id, self.id2intent, tag2id, id2tag = d.load_prepare_dataset(curdir)
        
        self.intent_train = df_train["intent_label"].map(self.intent2id).values
        self.intent_valid = df_valid["intent_label"].map(self.intent2id).values
        self.intent_test = df_test["intent_label"].map(self.intent2id).values

        self.encoded_train = encode_dataset(tokenizer, df_train["words"], max_length)
        self.encoded_valid = encode_dataset(tokenizer, df_valid["words"], max_length)
        self.encoded_test = encode_dataset(tokenizer, df_test["words"], max_length)


    def train(self, epochs, batch_size):

        time = datetime.datetime.now()
        name = f"intents_cls_e{epochs}_bs{batch_size}"

        checkpoint_path = os.path.join(self.model_save_dir, name)

        # Create a callback that saves the model's weights
        cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                        save_weights_only=True,
                                        verbose=1)


        history = self.model.fit(self.encoded_train,
                        self.intent_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(self.encoded_valid, self.intent_valid),
                        callbacks=cp_callback)



if __name__ == "__main__":
    curdir = Path(__file__).parent.absolute()
    trainer = ModelTrainer(args.max_length,\
         curdir, bert_model_name=args.model_name, dset_name=args.dataset)
    trainer.train(epochs=2, batch_size=16)
    model = trainer.model
    del trainer
    print(model)