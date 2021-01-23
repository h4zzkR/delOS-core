import torch.nn as nn
import torch.nn.functional as F

class DenseClassifierModel(nn.Module):
    """
    Simple network for intent classification
    """
    def __init__(self, embedding_dim=768, intent_num_labels=2, dropout_prob=0.15, from_logits=True):
        super().__init__()
        self.from_logits = from_logits
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(embedding_dim, intent_num_labels)
        self.act = F.softmax

    def forward(self, x):
        inputs = self.dropout(x)
        inputs = self.classifier(inputs)
        if not self.from_logits:
            inputs = self.act(inputs)
        return inputs