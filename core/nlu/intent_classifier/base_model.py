import tensorflow as tf

class BaseClassifierModel(tf.keras.Model):
    def __init__(self):
        super().__init__(name="intent_classifier")
        pass

    def call(self, inputs, **kwargs):
        pass