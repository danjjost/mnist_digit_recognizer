from config import Config
from src.neuralnet.network import Network
from src.pipeline.population import PopulationDTO


class TopPercentileSelector():
    def __init__(self, config: Config):
        self.percentile = config.top_percentile
        
    def select(self, population: PopulationDTO) -> list[Network]:
        population.population.sort(key=lambda network: network.score, reverse=True)
        
        max_element = int(len(population.population) * self.percentile)
        if max_element == 0:
            max_element = 1
            
        top_percentile = population.population[:max_element]
        
        return top_percentile