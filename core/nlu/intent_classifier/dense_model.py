from .base_model import BaseClassifierModel
import tensorflow as tf
from ..featurizers.transformer_featurizer import SenteceFeaturizer
from tensorflow.keras.layers import Dropout, Dense, GlobalAveragePooling1D
from tensorflow.keras.activations import softmax


class DenseClassifierModel(BaseClassifierModel):

    def __init__(self, intent_num_labels=None, dropout_prob=0.15):
        super().__init__()
        self.dropout = Dropout(dropout_prob)
        self.intent_classifier = Dense(intent_num_labels)

    def call(self, ft_inputs, **kwargs):
        ft_inputs = self.dropout(ft_inputs)
        out = self.intent_classifier(ft_inputs)
        out = softmax(out)
        return out