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

class FolderIndex(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def walker(self, path):
        dirs = [os.path.join(path, i) for i in os.listdir(path)]
        for i in dirs:
            if '#' not in i.split('/')[-1]:
                if '.txt' not in i:
                    i = os.path.join(i, 'texts')
                    for path, subdirs, files in os.walk(i):
                        for name in files:
                            # if 'PROC' not in name:
                            yield os.path.join(path, name)
                else:
                    yield i

    def __iter__(self):
        for fname in self.walker(self.dirname):
            for line in open(fname):
                yield line.split()



## TODO REDO
class EmbeddingTrainer():
    def __init__(self, train_type='gensim', gensim_type='fasttext', train_dir=None,
                window_size=6, emb_dim=256, min_count=10, epochs=15, 
                num_workers=None):
        self.train_type = train_type
        if self.train_type == 'gensim':
            self.embedding_type = gensim_type
        else:
            self.embedding_type = train_type
        self.train_dir = train_dir
        self.window = window_size
        self.embedding_dim = emb_dim
        self.min_count = min_count
        self.epochs = epochs

        if num_workers is None:
            self.num_workers = os.cpu_count() - 1
        else:   self.num_workers=num_workers

        self.logger = init_logger(__name__, testing_mode=False)
        self.name, self.model_name = self.get_name()
        self.data_loader = FolderIndex(self.train_dir)

        self.MODELS_PATH = os.path.join(ROOT_DIR, 'data/embedder_models')

    def get_name(self):
        name = f'{self.embedding_type.upper()}_SIZE_{self.embedding_dim}_WIN_{self.window}_EPOCH_{str(self.epochs)}'
        model_name = name + '.model'
        return name, model_name

    def init_model(self):
        if self.embedding_type == 'w2v':
            self.model = Word2Vec(size=self.embedding_dim, window=self.window, min_count=self.min_count,
                        workers=self.num_workers, iter=1)  # instantiate
        elif self.embedding_type == 'fasttext':
            self.model = FastText(size=self.embedding_dim, window=self.window, min_count=self.min_count,
                            workers=self.num_workers, iter=1)  # instantiate
        self.logger.info("Building vocabulary from corpus...")
        start_time = time.time()
        self.model.build_vocab(sentences=self.data_loader)
        self.logger.info("Vocabulary built in --- %s sec ---" % (time.time() - start_time))
        self.logger.info("Model is ready for training.")


    def train_model(self):
        start_time = time.time()
        self.logger.info(f"Starting {self.name} model training...")
        # try:
        self.model.train(self.data_loader, total_examples=self.model.corpus_count, epochs=self.epochs)
        self.logger.info("Model trained in --- %s sec ---" % (time.time() - start_time))
        # except Exception as err:
        #     self.logger.error(err)

    def extract_vectors(self, directory, bin):
        if bin:
            name = get_tmpfile(os.path.join(directory, self.name + '.bin'))
        else:
            name = get_tmpfile(os.path.join(directory, self.name + '.kv'))
        self.model.wv.save(name, separately=[])
        return name

    def save_model(self, key_vectors=True, bin_vectors=True):
        dtime = datetime.datetime.now()
        dtime = f"{self.embedding_type.upper()}.{dtime.year}.{dtime.month}.{dtime.day}_{dtime.hour}-{dtime.minute}-{dtime.second}"
        directory = os.path.join(self.MODELS_PATH, dtime)
        Path(directory).mkdir(parents=True, exist_ok=True)
        path = get_tmpfile(os.path.join(directory, self.model_name))

        if key_vectors is True:
            name = self.extract_vectors(directory, bin_vectors)
        else:
            self.model.save(path, separately=[])
        self.keyed_model = key_vectors
        self.logger.info(f"Model {self.model_name} saved")