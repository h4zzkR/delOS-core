import os
import datetime
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer
from pathlib import Path
from tensorflow.keras.optimizers import Adam
from backend.backbones.nlu_unit import ModuleUnit
from backend.functional import tf_set_memory_growth
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from backend.config import ROOT_DIR, TAGGER_DIR, NLU_CONFIG, MODELS_FIT_PARAMS
from backend.core.nlu.semantic_tagger.dense_model import DenseTagsExtractor
from backend.core.nlu.tools.utils import TagsDatasetLoader as DatasetLoader
from backend.core.nlu.featurizers.transformer_featurizer import SentenceFeaturizer
from backend.core.nlu.tools.utils import encode_dataset, encode_token_labels, jsonread, \
    load_tags_map, jsonread, load_map, dump

class SemanticTagger(ModuleUnit):
    def __init__(self, module_name = 'semantic_tagger'):
        super().__init__(module_name)
        self.intent = None
        self.base_model_ = 'semantic_tagger_base_model'
        self.models_dir = os.path.join(TAGGER_DIR, 'models')

    def _load_base_model(self, out_dim):
        if str(NLU_CONFIG[self.base_model_]) == 'dense':
            if self.fit_params['from_logits']:
                model = DenseTagsExtractor(out_dim, from_logits=True)
            else:
                model = DenseTagsExtractor(out_dim, from_logits=False)
        else:
            return IndentationError
        return model

    def _encode_tokenize(self, ft, seq):
        encoded = encode_dataset(ft, seq).tolist()
        encoded = ft.encode(
            encoded, output_value='token_embeddings', convert_to_numpy=True, is_pretokenized=True
        )
        return np.array(encoded)

    def _load_format_dataset(self, dataset, intent=None):
        d = DatasetLoader(dataset)
        featurizer = SentenceFeaturizer()
        df_train, intent2id, id2intent, self.tag2id, self.id2tag = d.load_prepare_dataset()
        self.fit_params['dataset_name'] = dataset

        if intent and intent in intent2id.keys():
            df_train = df_train[df_train['intent_label'] == intent]

        encoded_valid, y_valid = None, None
        if self.fit_params['validate']:
            validate_prob = self.fit_params['validate_prob']
            length = len(df_train)
            df_valid, df_train = df_train.loc[:int(validate_prob * length)], df_train.loc[int(validate_prob * length):]
            encoded_valid = self._encode_tokenize(featurizer, df_valid['words'])
            y_valid = encode_token_labels(df_valid["words"], df_valid["word_labels"], featurizer, self.tag2id)

        encoded_train = self._encode_tokenize(featurizer, df_train['words'])
        y_train = encode_token_labels(df_train["words"], df_train["word_labels"], featurizer, self.tag2id)

        self.fit_params['output_length'] = len(self.tag2id)
        return (encoded_train, y_train), (encoded_valid, y_valid)

    def _load_fit_parameters(self):
        self.fit_params = jsonread(MODELS_FIT_PARAMS)[self.base_model_]
        self.fit_params['validate'] = True
        if self.fit_params['validate_prob'] is None or self.fit_params['validate_prob'] == 1:
            self.fit_params['validate'] = False

    def load(self, intent):
        self.intent = intent
        checkpoint_path = os.path.join(self.models_dir, f"{self.intent}")
        try:
            ckp = Path(os.path.join(checkpoint_path, 'checkpoint')).read_text()
            m = ckp.splitlines()[0].split(': ')[-1]
            self.fit_params = jsonread(os.path.join(checkpoint_path , 'params.json'))
            checkpoint_path = os.path.join(checkpoint_path, m[1:-1])
            # _, self.id2tag = load_map(os.path.join(self.fit_params["dataset_name"], 'vocab.tag'))

            self.model = self._load_base_model(self.fit_params["output_length"])
            self.model.load_weights(checkpoint_path)
        except FileNotFoundError:
            print(f'{intent} tagger model not fitted, needs training')
            self.intent = None

    def fit(self, dataset, intent):
        """
        Dataset name, intent class
        """
        self.intent = intent
        self._load_fit_parameters()
        (x_train, y_train), (x_valid, y_valid) = self._load_format_dataset(dataset)
        self.model = self._load_base_model(self.fit_params['output_length'])

        opt = Adam(learning_rate=self.fit_params['learning_rate'], epsilon=self.fit_params['epsilon'])
        if self.fit_params['from_logits']:
            loss = SparseCategoricalCrossentropy(from_logits=True)
        else:
            loss = SparseCategoricalCrossentropy(from_logits=False)
        metrics = SparseCategoricalAccuracy('accuracy')
        self.model.compile(optimizer=opt, loss=loss, metrics=metrics)

        time = datetime.datetime.now()

        if isinstance(self.intent, list):
            self.intent = '@'.join(self.intent)

        checkpoint_path = os.path.join(self.models_dir, f"{self.intent}")
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        cp_callback = ModelCheckpoint(filepath=os.path.join(checkpoint_path, f"{self.intent}_model"), \
                                        save_weights_only=True, verbose=1)
        if self.fit_params['validate']:
            history = self.model.fit(x_train, y_train,
                        epochs=self.fit_params['epochs'], batch_size=self.fit_params['batch_size'],
                        validation_data=(x_valid, y_valid), callbacks=cp_callback)
        else:
            history = self.model.fit(x_train, y_train,
                        epochs=self.fit_params['epochs'], batch_size=self.fit_params['batch_size'],
                        callbacks=cp_callback)
        # LOGGING
        print(f"{self.intent} tagger fitted")
        dump(self.fit_params, os.path.join(checkpoint_path, 'params.json'))
        

    def tag(self, inputs):
        # TODO: extend this with decode_predictions code
        out = np.squeeze(np.squeeze(self.model.predict(inputs), 0), 0)
        out = np.argmax(out, 1)[1:-1]
        return out

    def fitted(self):
        return True if self.intent else False


if __name__ == "__main__":
    tf_set_memory_growth()
    obj = SemanticTagger()
    obj.fit('data/nlu_data/custom', 'intent')
    # obj.load('intent')