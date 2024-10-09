import unittest

from src.digit_recognition.convolution.feature_mapper import FeatureMapper
from src.digit_recognition.convolution.kernel import Kernel


class TestFeatureMapper(unittest.TestCase):
    
    def test_generates_correct_feature_map(self):
        two_dimensional_array: list[list[float]] = [
            [1,3,2,3,1],
            [4,5,2,3,2],
            [3,4,2,4,6],
            [1,3,2,2,4],
            [4,5,5,1,3]
        ]
        
        kernel = Kernel([
            [2,4,6],
            [2,2,4],
            [5,1,3]
        ])
        
        output = FeatureMapper().map(kernel, two_dimensional_array)
        
        assert output.map[0][0] == 77
        assert output.map[2][2] == 115
        assert output.map[0][2] == 72