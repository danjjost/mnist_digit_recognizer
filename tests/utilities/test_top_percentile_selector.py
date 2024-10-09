import unittest

from config import Config
from src.neuralnet.network import Network
from src.pipeline.population import PopulationDTO
from src.utilities.top_percentile_selector import TopPercentileSelector


class TestTopPercentileSelector(unittest.TestCase):

    def test_returns_only_the_top_percent(self):
        network1 = Network([1,1])
        network1.score = 1
        
        network2 = Network([1,1])
        network2.score = 2
        
        network3 = Network([1,1])
        network3.score = 3
        
        network4 = Network([1,1])
        network4.score = 4
        
        population = PopulationDTO([network1, network2, network3, network4])
        
        
        config = Config()
        config.top_percentile = 0.50
        
        
        top_percentile = TopPercentileSelector(config).select(population)
        

        assert network3 in top_percentile
        assert network4 in top_percentile
    
    def test_returns_top_one_if_percentile_rounds_down_to_zero(self):
        
        network1 = Network([1,1])
        network1.score = 1
        
        network2 = Network([1,1])
        network2.score = 2
        
        network3 = Network([1,1])
        network3.score = 3
        
        network4 = Network([1,1])
        network4.score = 4
        
        population = PopulationDTO([network1, network2, network3, network4])
        
        
        config = Config()
        config.top_percentile = 0.10
        
        
        top_percentile = TopPercentileSelector(config).select(population)
        

        assert len(top_percentile) == 1
        assert network4 in top_percentile