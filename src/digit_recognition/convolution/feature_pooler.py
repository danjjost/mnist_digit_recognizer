from src.digit_recognition.convolution.feature_mapper import FeatureMap


class FeaturePooler():
    def pool_maps(self, feature_map: list[FeatureMap]):
        return [self.pool_map(map) for map in feature_map]
    
    def pool_map(self, feature_map: FeatureMap):
        pooled_feature_map_dimension = len(feature_map.map) // 2
        pooled_feature_map: list[list[float]] = [[0 for _ in range(pooled_feature_map_dimension)] for _ in range(pooled_feature_map_dimension)]
        
        for x in range(pooled_feature_map_dimension):
            for y in range(pooled_feature_map_dimension):
                pooled_feature_map[x][y] = self.pool_element(feature_map.map, x, y)
                
        return FeatureMap(pooled_feature_map)    
            
    def pool_element(self, feature_map: list[list[float]], x_origin: int, y_origin: int) -> float:
        all_elements: list[float] = []
        
        for x in range(2):
            for y in range(2):
                all_elements.append(feature_map[x + x_origin][y + y_origin])
                
        return max(all_elements)