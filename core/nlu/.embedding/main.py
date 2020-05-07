import sys, os; sys.path.insert(0,'../chloeAI')
from gensim.corpora.textcorpus import TextCorpus
from gensim.test.utils import datapath
from gensim import utils
from gensim.test.utils import get_tmpfile
from gensim.models import Word2Vec, FastText
from gensim.models import KeyedVectors
from gensim.models.fasttext import FastTextKeyedVectors
import multiprocessing 
import time, datetime
from pathlib import Path

from nltk import ngrams
import numpy as np
from itertools import chain

from settings import *


class EmbeddingLoader():
    def __init__(self, model=None, workers=1):
        """
        Класс для загрузки моделей (векторов) embedding.
        Путь к модели бинарника векторов (или просто модели,
        но тогда явно указать key_vectors=False).
        Явно указать класс embedding (fasttext/w2v).
        workers.
        Также в будущем может быть будут использоваться
        вектора из Transformer моделей (контекстные эмбдд).
        embedvec/embed

        Рекомендовано использовать fasttext из-за способности
        обрабатывать и строить вектора для слов, которых нет в
        словаре.
        """
        # FASTTEXT_SIZE_128_WIN_6_EPOCH_5.bin
        self.model_path = model
        if 'gens' in model:
            self.embedding_type = 'gensim'
        else:
            self.embedding_type = 'other'
        if 'w2v' in model:
            self.gensim_type = 'w2v'
        elif 'ftt' in model:
            self.gensim_type = 'ftt'
        if 'embedvec' in model:
            self.keyed_model = True
        else:
            self.keyed_model = Falses
        self.load()

    def load(self):
        if not self.keyed_model:
            self.load_model()
        else:
            self.load_vectors()
                    
    def parse_name(self):
        idx = self.model_path.find('dim_')
        part = self.model_path[idx+4:]
        self.emb_dim = int(part.split('_')[0])
        self.name = self.model_path.split('/')[-1]

    def load_vectors(self):
        self.parse_name()
        start_time = time.time()
        if self.gensim_type == 'w2v':
            self.model = KeyedVectors.load_word2vec_format(self.model_path, binary=True)
        elif self.gensim_type == 'ftt':
            self.model = FastTextKeyedVectors.load(self.model_path)
        LOGGER.info(f"{self.name} loaded in --- {time.time() - start_time } sec ---")
        self.gensim_type = type

    def load_model(self, type='fasttext'):
        self.parse_name()
        start_time = time.time()
        if self.gensim_type == 'w2v':
            self.model = Word2Vec.load(self.model_path)
        elif type == 'ftt':
            self.model = FastText.load(self.model_path, binary=True)
        LOGGER.info(f"{self.name} loaded in --- {time.time() - start_time } sec ---")
        self.gensim_type = type


class Embeddings():
    """
    Class for embedding control and run.
    Use loaded or trained model here.
    """
    def __init__(self, loader_model):
        self.model = loader_model.model
        self.keyed_model = loader_model.key_vectors
        self.emb_dim = loader_model.emb_dim



# if __name__ == "__main__":
#     e = EmbeddingLoader(model='/mnt/sdb1/lab/chloeAI/storage/gens_w2v_embedvec_win_5_dim_64_default.bin')


### CHANGELOG
# LOADING W2V: WORKS
# LOADING FTT: NS
# LOADING MODELS: NS