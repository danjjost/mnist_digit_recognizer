from typing import Any, Optional

from src.digit_recognition.convolution.feature_mapper import FeatureMap, FeatureMapper
from src.digit_recognition.convolution.feature_pooler import FeaturePooler
from src.digit_recognition.convolution.kernel import Kernel


class ImageConvolver():
    def __init__(self, feature_mapper: Optional[FeatureMapper] = None, feature_pooler: Optional[FeaturePooler] = None) -> None:
        self.feature_mapper = feature_mapper or FeatureMapper()
        self.feature_pooler = feature_pooler or FeaturePooler()
        self.default_kernels: list[Kernel] = [
            Kernel([
                [-0.03431337,  0.23040098, -0.32356027],
                [-0.29984933, -0.107909,   -0.09893321],
                [-0.3094766,   0.05762419,  0.16151679]
            ]),
            Kernel([
                [0.14889278, 0.30925864, 0.06695294],
                [1.0256958,  0.72835135, 0.9281884 ],
                [1.1492832,  0.96711564, 0.55681336]
            ]),
            Kernel([
                [-1.4979255,  -1.0722408,  -1.1007311 ],
                [-0.6175827,  -0.31201962, -1.0679379 ],
                [ 1.0477579,   1.0510583,   0.7730589 ]
            ]),
            Kernel([
                [-0.19065571,  0.74855924,  1.3533834 ],
                [ 0.12472851,  1.5501034,   1.3432286 ],
                [ 0.08097316,  0.28835735, -0.19931725]
            ]),
            Kernel([
                [ 1.6856706,  -0.3631743,  -0.78715235],
                [ 0.25925413, -1.0696309,  -0.20901743],
                [-0.9097826,  -0.0019105,   1.3699291 ]
            ]),
            Kernel([
                [ 0.82920474, -0.7190052,  -1.5637227 ],
                [ 1.3368105,  -0.02833749, -1.4797611 ],
                [ 1.2175187,   1.092427,    0.26224104]
            ]),
            Kernel([
                [-1.3112082,  -1.8706939,  -1.5997347 ],
                [ 0.8280369,   0.09072771, -0.39141402],
                [ 1.5034438,   1.8352776,   1.432895  ]
            ]),
            Kernel([
                [-1.3916092,  -1.4993192,  0.00945535],
                [-1.530441,   -0.16645603,  1.4911125 ],
                [-0.80281365,  1.0331721,   1.5699245 ]
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