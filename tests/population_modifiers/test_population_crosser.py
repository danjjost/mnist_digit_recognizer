import unittest
from unittest.mock import MagicMock

from config import Config
from src.pipeline.network_crosser import NetworkCrosser
from src.pipeline.population_generator import PopulationGenerator
from src.pipeline.population_modifiers.population_crosser import PopulationCrosser
from src.utilities.number_of_crosses_calculator import NumberOfCrossesCalculator


class TestPopulationCrosser(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config()

        self.default_number_of_crosses = 1

        self.network_crosser = MagicMock(spec=NetworkCrosser)
        self.number_of_crosses_calculator = MagicMock(spec=NumberOfCrossesCalculator)
        self.number_of_crosses_calculator.get_number_of_crosses.return_value = self.default_number_of_crosses
        
        
    def test_calls_network_crosser_for_each_network(self):
        population = PopulationGenerator().generate(3, [1, 1])
        
        population_crosser = PopulationCrosser(Config(), self.network_crosser, self.number_of_crosses_calculator)
    
    
        population_crosser.run(population)
    
    
        assert self.network_crosser.cross.call_count == 3
        assert self.network_crosser.cross.call_args_list[0][0][0] == population.population[0]
        assert self.network_crosser.cross.call_args_list[1][0][0] == population.population[1]
        assert self.network_crosser.cross.call_args_list[2][0][0] == population.population[2]