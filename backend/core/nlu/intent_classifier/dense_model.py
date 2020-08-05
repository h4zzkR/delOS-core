from .base_model import BaseClassifierModel
import tensorflow as tf
from backend.core.nlu.featurizers.transformer_featurizer import SentenceFeaturizer
from tensorflow.keras.layers import Dropout, Dense, GlobalAveragePooling1D
from tensorflow.keras.activations import softmax


class DenseClassifierModel(BaseClassifierModel):

    def __init__(self, intent_num_labels=None, dropout_prob=0.15, from_logits=True):
        super().__init__()
        self.from_logits = from_logits
        self.dropout = Dropout(dropout_prob)
        self.intent_classifier = Dense(intent_num_labels)

    def call(self, ft_inputs, **kwargs):
        ft_inputs = self.dropout(ft_inputs)
        out = self.intent_classifier(ft_inputs)
        if not self.from_logits:
            out = softmax(out)
        return out

# class DenseClassifierTorchModel(nn.Module):
#     def __init__(self, intent_num_labels=None, dropout_prob=0.15, from_logits=True):
#         super(self).__init__()
#         self.drop = nn.Dropout(0.2)
#         self.out = nn.Linear(128, 10)
#         self.act = nn.ReLU()

#     def forward(self, x):
#         x = self.act(self.conv(x)) # [batch_size, 28, 26, 26]
#         x = self.pool(x) # [batch_size, 28, 13, 13]
#         x = x.view(x.size(0), -1) # [batch_size, 28*13*13=4732]
#         x = self.act(self.hidden(x)) # [batch_size, 128]
#         x = self.drop(x)
#         x = self.out(x) # [batch_size, 10]
#         return x