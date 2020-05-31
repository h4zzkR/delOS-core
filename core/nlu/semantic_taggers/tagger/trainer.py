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
from ...nlu_data.utils import DatasetLoader, encode_dataset, encode_token_labels
from transformers import BertTokenizer
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.client import device_lib
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from .base import SemanticTagsExtractor


parser = argparse.ArgumentParser(description='Trainer for bert-based intent classificator')
parser.add_argument('--model-name', type=str, default='bert-base-cased')
# solve problem with max length

parser.add_argument('--max_length', type=int, default=49, help='Max tokens in sequence')
parser.add_argument('--dataset', type=str, default='merged', help='Choose dataset dir for training')


args = parser.parse_args()


class TaggerTrainer():

    def __init__(self, max_length, curdir, name='tag_extractor', dset_name='merged',
                 bert_model_name="bert-base-cased", model_save_dir='model'):

        self.training_model = name
        tokenizer = BertTokenizer.from_pretrained(bert_model_name)

        self.model_save_dir = os.path.join(curdir, model_save_dir)
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

        self.dataset_preload(dset_name, curdir, tokenizer, max_length)

        self.model = SemanticTagsExtractor(
                    tag_num_labels=len(self.tag_map),
                    bert_model=bert_model_name,
                    )

        opt = Adam(learning_rate=3e-5, epsilon=1e-08)
        loss = SparseCategoricalCrossentropy(from_logits=True)
        metrics = SparseCategoricalAccuracy('accuracy')

        self.model.compile(optimizer=opt, loss=loss, metrics=metrics)


    def dataset_preload(self, dset_name, curdir, tokenizer, max_length):
        d = DatasetLoader(dset_name)
        df_train, df_valid, df_test, \
                intent2id, id2intent, self.tag_map = d.load_prepare_dataset(curdir)

        print('Encoding data...')
        self.encoded_train = encode_dataset(tokenizer, df_train["words"], max_length)
        self.encoded_valid = encode_dataset(tokenizer, df_valid["words"], max_length)
        self.encoded_test = encode_dataset(tokenizer, df_test["words"], max_length)
        
        self.tag_train = encode_token_labels(
            df_train["words"], df_train["word_labels"], tokenizer, self.tag_map, max_length)
        self.tag_valid = encode_token_labels(
            df_valid["words"], df_valid["word_labels"], tokenizer, self.tag_map, max_length)
        self.tag_test = encode_token_labels(
            df_test["words"], df_test["word_labels"], tokenizer, self.tag_map, max_length)


    def train(self, epochs, batch_size):
        time = datetime.datetime.now()
        name = f"tags_extractor_e{epochs}_bs{batch_size}"

        checkpoint_path = os.path.join(self.model_save_dir, name)

        # Create a callback that saves the model's weights
        cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                        save_weights_only=True,
                                        verbose=1)


        history = self.model.fit(self.encoded_train, self.tag_train,
                        epochs=epochs, batch_size=batch_size,
                        validation_data=(self.encoded_valid, self.tag_valid),
                        callbacks=cp_callback)
        return self.model





class ModelTrainerD():
    def __init__(self, model_name, max_length, curdir):
        df_train, df_valid, df_test, intent_names, \
                self.intent2id, self.id2intent, self.slot_map = load_prepare_dataset(curdir)
        # Y's:
        self.intent_train = df_train["intent_label"].map(self.intent2id).values
        self.intent_valid = df_valid["intent_label"].map(self.intent2id).values
        self.intent_test = df_test["intent_label"].map(self.intent2id).values

        tokenizer = BertTokenizer.from_pretrained(model_name)
        self.curdir = curdir
        # X's:
        print('Encoding data...')
        self.encoded_train = encode_dataset(tokenizer, df_train["words"], max_length)
        self.encoded_valid = encode_dataset(tokenizer, df_valid["words"], max_length)
        self.encoded_test = encode_dataset(tokenizer, df_test["words"], max_length)

        self.slot_train = encode_token_labels(
            df_train["words"], df_train["word_labels"], tokenizer, self.slot_map, max_length)
        self.slot_valid = encode_token_labels(
            df_valid["words"], df_valid["word_labels"], tokenizer, self.slot_map, max_length)
        self.slot_test = encode_token_labels(
            df_test["words"], df_test["word_labels"], tokenizer, self.slot_map, max_length)


        self.intent_model = SlotIntentDetectorModelBase(
                    intent_num_labels=len(self.intent_map),
                    slot_num_labels=len(self.slot_map)
                    )

        opt = Adam(learning_rate=3e-5, epsilon=1e-08)
        losses = [SparseCategoricalCrossentropy(from_logits=True),
                  SparseCategoricalCrossentropy(from_logits=True)]

        metrics = [SparseCategoricalAccuracy('accuracy')]
        self.intent_model.compile(optimizer=opt, loss=losses, metrics=metrics)

    def train(self, epochs, batch_size, model_save_dir='model'):
        model_save_dir = os.path.join(self.curdir, model_save_dir)
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        time = datetime.datetime.now()
        name = f"intents_cls_e{epochs}_bs{batch_size}"

        checkpoint_path = os.path.join(model_save_dir, name)
        checkpoint_dir = os.path.dirname(checkpoint_path)

        # Create a callback that saves the model's weights
        cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                                        save_weights_only=True,
                                                        verbose=1)


        history = self.intent_model.fit(self.encoded_train, (self.slot_train,
                        self.intent_train), epochs=epochs, batch_size=batch_size,
                        validation_data=(self.encoded_valid,
                        (self.slot_valid, self.intent_valid)), callbacks=cp_callback)

        return self.intent_model








if __name__ == "__main__":
    # python -m core.nlu.semantic_taggers.tagger.trainer
    curdir = Path(__file__).parent.absolute()
    trainer = TaggerTrainer(args.max_length,\
         curdir, bert_model_name=args.model_name, dset_name=args.dataset)
    trainer.train(epochs=2, batch_size=16)
    model = trainer.model
    del trainer
    print(model)