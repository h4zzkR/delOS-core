import tensorflow as tf

class SemanticTagsExtractor(tf.keras.Model):
    def __init__(self):
        super().__init__(name='semantic_tag_extractor')
        pass

    def fit(self, dataset, intent):
        pass

    def call(self, inputs, **kwargs):
        pass
