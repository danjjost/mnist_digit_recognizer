from typing import Any, Optional

from src.digit_recognition.convolution.feature_mapper import FeatureMap, FeatureMapper
from src.digit_recognition.convolution.feature_pooler import FeaturePooler
from src.digit_recognition.convolution.kernel import Kernel


class ImageConvolver():
    def __init__(self, feature_mapper: Optional[FeatureMapper] = None, feature_pooler: Optional[FeaturePooler] = None) -> None:
        self.feature_mapper = feature_mapper or FeatureMapper()
        self.feature_pooler = feature_pooler or FeaturePooler()
        self.default_kernels: list[Kernel] = [
            # Horizontal Sobel Filter
            Kernel([
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ]),
            # Vertical Sobel Filter
            Kernel([
                [-1, -2, -1],
                [0,  0,  0],
                [1,  2,  1]
            ]),
            # Laplacian Filter
            Kernel([
                [0, -1,  0],
                [-1, 4, -1],
                [0, -1,  0]
            ]),
            # Gaussian Blur Filter (normalized)
            Kernel([
                [1/16, 2/16, 1/16],
                [2/16, 4/16, 2/16],
                [1/16, 2/16, 1/16]
            ]),
            # Diagonal Sobel Filter (optional)
            Kernel([
                [2, 1, 0],
                [1, 0, -1],
                [0, -1, -2]
            ]),
            # Anti-Diagonal Sobel Filter (optional)
            Kernel([
                [0,  1,  2],
                [-1, 0,  1],
                [-2, -1, 0]
            ]),
            Kernel([
                [0, 0, -1, 0, 0],
                [0, -1, -2, -1, 0],
                [-1, -2, 16, -2, -1],
                [0, -1, -2, -1, 0],
                [0, 0, -1, 0, 0]
            ]),Kernel([
                [-1, -1, -1],
                [-1, 8, -1],
                [-1, -1, -1]
            ])
        ]
        
    def convolve(self, image_array: list[float]) -> list[float]:
        two_dimensional_array = self.to_2d_array(image_array) 
        
        feature_maps: list[FeatureMap] = []
        for kernel in self.default_kernels:
            feature_maps.append(self.feature_mapper.map(kernel, two_dimensional_array))
        
        feature_maps = self.feature_pooler.pool_maps(feature_maps)
        
        return self.flatten_to_single_array(feature_maps)

    def flatten_to_single_array(self, feature_maps: list[FeatureMap]) -> list[float]:
        return self.flatten(self.flatten([feature_map.map for feature_map in feature_maps]))
    
    def to_2d_array(self, image_array: list[float]) -> list[list[float]]:
        dimensions = int(len(image_array) ** 0.5)
        
        two_dimensional_image_array: list[list[float]] = [[0] * dimensions for _ in range(dimensions)]
        
        for i in range(dimensions):
            for j in range(dimensions):
                two_dimensional_image_array[i][j] = image_array[i * dimensions + j]
                
        return two_dimensional_image_array
    
    def flatten(self, two_dimensional_array: list[list[Any]]) -> list[Any]:
        return [element for row in two_dimensional_array for element in row]