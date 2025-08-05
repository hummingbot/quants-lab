from core.features.feature_base import FeatureBase


class DataStructureBase:
    def __init__(self, data, *args, **kwargs):
        self.data = data

    def add_feature(self, feature: FeatureBase):
        self.data = feature.calculate(self.data)
        return self

    def add_features(self, features: list[FeatureBase]):
        for feature in features:
            self.data = feature.calculate(self.data)
        return self
