import tensorflow as tf
from transformers import TFBertModel, BertTokenizer

class BertBase():
    def __init__(model_name="bert-base-cased"):
        self.bert = TFBertModel.from_pretrained(model_name)