import tensorflow as tf
from transformers import TFBertModel, BertTokenizer
from tensorflow.keras.layers import Dropout, Dense, GlobalAveragePooling1D


class SemanticTagsExtractor(tf.keras.Model):
    def __init__(self, bert_model, tag_num_labels=None, dropout_prob=0.1):
        super().__init__(name='tag_extractor')
        if isinstance(bert_model, str):
            self.bertbase = TFBertModel.from_pretrained(bert_model)
        else:
            self.bertbase = bert_model
        self.dropout = Dropout(dropout_prob)

        self.tag_extractor = Dense(tag_num_labels, name="tag_extractor")

    def call(self, inputs, **kwargs):
        sequence_output, pooled_output = self.bertbase(inputs, **kwargs)
        sequence_output = self.dropout(sequence_output,
                                       training=kwargs.get("training", False))
        tag_logists = self.tag_extractor(sequence_output)
        return tag_logists
