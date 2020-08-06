from .base_model import SemanticTagsExtractor
import tensorflow as tf
from transformers import TFBertModel, BertTokenizer
from tensorflow.keras.layers import Dropout, Dense, GlobalAveragePooling1D
from tensorflow.keras.activations import softmax
import numpy as np

from future.utils import with_metaclass


class DenseTagsExtractor(tf.keras.Model):
    def __init__(self, tag_num_labels=None, dropout_prob=0.15, from_logits=True):
        super().__init__()
        self.from_logits = from_logits
        self.dropout = Dropout(dropout_prob)
        self.tag_extractor = Dense(tag_num_labels, name="tag_extractor")

    def call(self, ft_inputs, **kwargs):
        ft_inputs = self.dropout(ft_inputs)
        out = self.tag_extractor(ft_inputs)
        if not self.from_logits:
            out = softmax(out)
        return out
