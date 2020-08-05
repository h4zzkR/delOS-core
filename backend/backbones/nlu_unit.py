import os
from abc import ABCMeta, abstractmethod, abstractproperty
from backend.config import MODELS_FIT_PARAMS

class ModuleUnit():
    def __init__(self, model_name, random_state=42):
        """
        Abstract class for semantic tagger, intent classifier and
        other nlu units
        """
        self.model_name = model_name
        self.random_state = random_state
        self.fit_params = None

    def build_out_map(self, dataset):
        pass

    def _load_fit_parameters(self):
        pass

    def load(self):
        pass

    def save(self):
        pass

    def _load_format_dataset(self, dataset):
        pass

    def fit(self):
        pass

    def fitted(self):
        pass

    def get_name(self):
        return self.model_name