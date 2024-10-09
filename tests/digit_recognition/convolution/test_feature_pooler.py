import numpy as np
from src.digit_recognition.convolution.feature_mapper import FeatureMap
from src.digit_recognition.convolution.feature_pooler import FeaturePooler


class TestFeaturePooler:
    def test_pooling(self):
        pooler = FeaturePooler()
        map_as_floats: list[list[float]] = [
            [1,2,3], 
            [4,5,6], 
            [7,8,9]
        ]
        feature_map = FeatureMap(map_as_floats)
        pooled = pooler.pool_map(feature_map)
        assert len(pooled.map) == 2
        assert len(pooled.map[0]) == 2
        assert np.array_equal(pooled.map, np.array([[5, 6], [8, 9]]))