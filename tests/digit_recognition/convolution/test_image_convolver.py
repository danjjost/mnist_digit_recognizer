from random import Random
import unittest
from unittest.mock import MagicMock, call

from src.digit_recognition.convolution.feature_mapper import FeatureMap, FeatureMapper
from src.digit_recognition.convolution.image_convolver import ImageConvolver

class TestImageConvolver(unittest.TestCase):
    def test_to_2d_array(self):
        image_array: list[float] = [0,1,2,3,4,5,6,7,8]
        two_dimensional_image_array = ImageConvolver().to_2d_array(image_array)
        
        assert two_dimensional_image_array == [[0,1,2],[3,4,5],[6,7,8]]
        
    def test_calls_feature_mapper_for_the_two_dimensional_array_and_each_default_kernel(self):
        feature_mapper = MagicMock(spec=FeatureMapper)
        image_convolver = ImageConvolver(feature_mapper)
        
        image_convolver.convolve([0,1,2,3,4,5,6,7,8])
        
        default_kernels = image_convolver.default_kernels
        
        expected_calls = [
            call(kernel, [[0, 1, 2], [3, 4, 5], [6, 7, 8]]) for kernel in default_kernels
        ]

        feature_mapper.map.assert_has_calls(expected_calls, any_order=True) 
        
    def test_flatten_can_flatten_a_list(self):
        some_list = [[1,2],[3,4],[5,6]]
        flattened_list =  ImageConvolver().flatten(some_list)
        
        assert flattened_list == [1,2,3,4,5,6]
        
    def test_convolve_returns_flattened_and_pooled_feature_maps(self):
        feature_mapper = MagicMock(spec=FeatureMapper)
        
        image_convolver = ImageConvolver(feature_mapper)
        default_kernels = image_convolver.default_kernels
        
        random_feature_maps = self.generate_random_feature_maps(len(default_kernels))
        feature_mapper.map.side_effect = random_feature_maps


        flattened_feature_maps = image_convolver.convolve([0,1,2,3,4,5,6,7,8])
        
        expected_flattened_feature_maps = [
            value for feature_map in random_feature_maps for row in feature_map.map for value in row
        ]
        
        pooled_flattened_feature_maps = self.get_pooled_flattened_feature_maps(expected_flattened_feature_maps)
        
        assert pooled_flattened_feature_maps == flattened_feature_maps
                        
    def get_pooled_flattened_feature_maps(self, flattened_feature_maps: list[float]) -> list[float]:
        pooled_flattened_feature_maps: list[float] = []
        for i in range(0, len(flattened_feature_maps), 4):
            pooled_flattened_feature_maps.append(max(flattened_feature_maps[i:i+4]))
        
        return pooled_flattened_feature_maps
    
    def generate_random_feature_maps(self, length: int) -> list[FeatureMap]:
        feature_maps: list[FeatureMap] = []
        
        for _ in range(length):
            feature_map: list[list[float]] = [
                [Random().random(), Random().random()],
                [Random().random(), Random().random()]
            ]
            
            feature_maps.append(FeatureMap(feature_map))
        
        return feature_maps
        