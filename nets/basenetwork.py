from abc import ABC, abstractmethod
import tensorflow as tf


class BaseNet(ABC):
    def __init__(self, name, num_classes, input_shape, pretrained_model=None):
        self._name = name
        self._pretrained_model = pretrained_model
        self._input_shape = input_shape
        self._num_classes = num_classes

    @property
    def pretrained_model(self):
        return self._pretrained_model

    @property
    def name(self):
        return self._name

    @abstractmethod
    def logits(self, input_batch):
        raise NotImplementedError

    @abstractmethod
    def input_shape(self):
        return self._input_shape

    @property
    def num_classes(self):
        return self._num_classes
