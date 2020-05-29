import tensorflow as tf
from transformers import TFBertModel, BertTokenizer
from tensorflow.keras.layers import Dropout, Dense, GlobalAveragePooling1D

# from ...bert.bertmodel import BertLayer

#TODO Split detection and extraction
# will use separate semantic taggers for various intents


class RequestIntentClassifier(tf.keras.Model):
    def __init__(self, bert_model, intent_num_labels=None, dropout_prob=0.1):
        super().__init__(name="intent_classificator")
        if isinstance(bert_model, str):
            self.bertbase = TFBertModel.from_pretrained(bert_model)
        else:
            self.bertbase = bert_model
        self.dropout = Dropout(dropout_prob)

        self.intent_classifier = Dense(intent_num_labels,
                                        name='intent_classifier')


    def call(self, inputs, **kwargs):
        sequence_output, pooled_output = self.bertbase(inputs, **kwargs)
        # The first output of the main BERT layer has shape:
        # (batch_size, max_length, output_dim)
        sequence_output = self.dropout(sequence_output,
                                    training=kwargs.get("training", False))

        # The second output of the main BERT layer has shape:
        # (batch_size, output_dim)
        # and gives a "pooled" representation for the full sequence from the
        # hidden state that corresponds to the "[CLS]" token.
        pooled_output = self.dropout(pooled_output,
                                    training=kwargs.get("training", False))
        intent_logits = self.intent_classifier(pooled_output)
        return intent_logits

# class SlotIntentDetectorModelBase(tf.keras.Model):

#     def __init__(self, intent_num_labels=None, slot_num_labels=None,
#                  model_name="bert-base-cased", dropout_prob=0.1):
#         super().__init__(name="joint_intent_slot")
#         self.bert = TFBertModel.from_pretrained(model_name)
#         self.dropout = Dropout(dropout_prob)
#         self.intent_classifier = Dense(intent_num_labels,
#                                        name="intent_classifier")
#         self.slot_classifier = Dense(slot_num_labels,
#                                      name="slot_classifier")

#     def call(self, inputs, **kwargs):
#         sequence_output, pooled_output = self.bert(inputs, **kwargs)

#         # The first output of the main BERT layer has shape:
#         # (batch_size, max_length, output_dim)
#         sequence_output = self.dropout(sequence_output,
#                                        training=kwargs.get("training", False))
#         slot_logits = self.slot_classifier(sequence_output)

#         # The second output of the main BERT layer has shape:
#         # (batch_size, output_dim)
#         # and gives a "pooled" representation for the full sequence from the
#         # hidden state that corresponds to the "[CLS]" token.
#         pooled_output = self.dropout(pooled_output,
#                                      training=kwargs.get("training", False))
#         intent_logits = self.intent_classifier(pooled_output)

#         return slot_logits, intent_logits

