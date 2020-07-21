import numpy as np
import tensorflow as tf
from sentence_transformers import SentenceTransformer

class SenteceFeaturizer():
    def __init__(self, model_name):
        self.featurizer = SentenceTransformer(model_name)
    
    def encode(self, inputs):
        return np.array(self.featurizer.encode(inputs))