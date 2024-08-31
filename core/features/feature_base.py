from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from pydantic import BaseModel

# Define a type variable that can be any subclass of FeatureConfig
T = TypeVar('T', bound='FeatureConfig')


class FeatureConfig(BaseModel):
    name: str


class FeatureBase(ABC, Generic[T]):
    def __init__(self, feature_config: T):
        self.config = feature_config

    @abstractmethod
    def calculate(self, data):
        """
        This method should be implemented by the child class to calculate the feature and add it to the data.
        :param data:
        :return:
        """
        ...
