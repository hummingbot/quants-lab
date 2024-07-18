from typing import List

from features.feature_base import FeatureBase


class DataHandlerBase:
    def __init__(self, data):
        self.data = data

    def add_feature(self, feature: FeatureBase):
        self.data = feature.calculate(self.data)
        return self

    def add_features(self, features: List[FeatureBase]):
        for feature in features:
            self.data = feature.calculate(self.data)
        return self
