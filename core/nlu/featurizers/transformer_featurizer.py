import numpy as np
import tensorflow as tf
from config import NLU_CONFIG
from sentence_transformers import SentenceTransformer, models

class SentenceFeaturizer():
    def __init__(self):
        self.featurizer  = SentenceTransformer(NLU_CONFIG['featurizer_model'])
        self.featurizer._modules['0'].max_seq_length = NLU_CONFIG['max_seq_length']
    
    def encode(self, inputs, **kwargs):
        return np.array(self.featurizer.encode(inputs, **kwargs))

    def tokenize(self, inputs):
        return np.array(self.featurizer.tokenize(inputs))