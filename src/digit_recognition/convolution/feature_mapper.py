

from src.digit_recognition.convolution.kernel import Kernel


class FeatureMap():
    def __init__(self, map: list[list[float]]) -> None:
        self.map = map
        
class FeatureMapper():
    def __init__(self) -> None:
        pass
    
    def map(self, kernel: Kernel, two_dimensional_array: list[list[float]]) -> FeatureMap:
        dimension = len(two_dimensional_array) - 2
        feature_map: list[list[float]] = [[0 for _ in range(dimension)] for _ in range(dimension)]
        
        for x in range(dimension):
            for y in range (dimension):
                feature_map[x][y] = self.get_element_wise_product(two_dimensional_array, kernel, x, y)
        
        return FeatureMap(feature_map)
                
    def get_element_wise_product(self, two_dimensional_array: list[list[float]], kernel: Kernel, x_origin: int, y_origin: int) -> float:
        total: float = 0
        for x in range(3):
            for y in range(3):
                total += two_dimensional_array[x + x_origin][y + y_origin] * kernel.schema[x][y]
                
        return total 