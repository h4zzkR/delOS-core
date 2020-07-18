import tensorflow as tf

class SemanticTagsExtractor(tf.keras.Model):
    def __init__(self):
        super().__init__(name='tag_extractor')
        pass

    def call(self, inputs, **kwargs):
        pass
