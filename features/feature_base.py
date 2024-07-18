from abc import ABC, abstractmethod


class FeatureBase(ABC):
    def __init__(self, **kwargs):
        self.params = kwargs

    @abstractmethod
    def calculate(self, data):
        """
        This method should be implemented by the child class to calculate the feature and add it to the data_handler.
        :param data_handler:
        :return:
        """
        ...
