import os
from abc import abstractmethod
from backend.functional import tf_set_memory_growth
# help text from snips nlu

class NLUEngine():
    """Abstraction which performs input intent parsing
    """
    def __init__(self):
        tf_set_memory_growth()

    @abstractmethod
    def fit(self, dataset, force_retrain):
        """Fit the intent classifier and taggers with a valid dataset

        Args:
            dataset (dict): valid dataset
            force_retrain (bool): specify whether or not sub units of the
            intent parser that may be already trained should be retrained
        """
        pass

    @abstractmethod
    def eval(self, dataset):
        """
        Switch to predict mode (load some modules which not needed in fit mode)

        Args:
            dataset (string): relative path to dataset in data/nlu_data
        """
        pass

    @abstractmethod
    def parse(self, text, top_n, intents=None):
        """Performs intent parsing on the provided *text*

        Args:
            text (str): input
            intents (str or list of str): if provided, reduces the scope of
                intent parsing to the provided list of intents
            top_n (int, optional): when provided, this method will return a
                list of at most top_n most likely intents, instead of a single
                parsing result.

        Returns:
            dict or list: the most likely intent(s) along with the extracted
            slots.
        """
        pass

    @abstractmethod
    def preprocess_text(self, text):
        pass

    @abstractmethod      
    def fitted(self):
        """Whether or not the intent classifier and taggers has already been fitted"""
        pass

    @abstractmethod
    def get_intents(self, text):
        """Performs intent classification on the provided *text* and returns
        the list of intents ordered by decreasing probability

        The length of the returned list is exactly the number of intents in the
        dataset + 1 for the None intent
        """
        pass

    @abstractmethod
    def build_result(self, intents_logits, tags_logits):
        """Tagger and classifier output to json-like data:
        {'intent' : name, 'tags' : {'a' : 'b'}}
        """
        pass

    @abstractmethod
    def get_tags(self, text, intent):
        """Extract slots from a text input, with the knowledge of the intent

        Args:
            text (str): input
            intent (str): the intent which the input corresponds to

        Returns:
            list: the list of extracted tags

        Raises:
            IntentNotFoundError: when the intent was not part of the training
                data
        """
        pass
