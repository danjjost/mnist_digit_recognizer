import unittest
from unittest.mock import MagicMock

from config import Config
from src.neuralnet.network import Network
from src.pipeline.population_modifiers.cross.network_crosser import NetworkCrosser
from src.pipeline.population_modifiers.cross.population_crosser import PopulationCrosser
from src.pipeline.population_modifiers.population_generator import PopulationGenerator
from src.pipeline.top_percentile_selector import TopPercentileSelector
from src.utilities.number_of_crosses_calculator import NumberOfCrossesCalculator


class TestPopulationCrosser(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config()

        self.default_number_of_crosses = 1

        self.network_crosser = MagicMock(spec=NetworkCrosser)
        self.number_of_crosses_calculator = MagicMock(spec=NumberOfCrossesCalculator)
        self.number_of_crosses_calculator.get_number_of_crosses.return_value = self.default_number_of_crosses
        self.top_percentile_selector = MagicMock(spec=TopPercentileSelector)
        
    def test_calls_network_crosser_for_each_network_and_a_network_from_the_top_percentile(self):
        top_network_1 = Network([1,1], self.config)
        top_network_2 = Network([1,1], self.config)
        self.top_percentile_selector.select.return_value = [top_network_1, top_network_2]
        
        population = PopulationGenerator().generate(3, [1, 1])
        
        population_crosser = PopulationCrosser(Config(), self.network_crosser, self.number_of_crosses_calculator, self.top_percentile_selector)
    
    
        population_crosser.run(population)
        
        self.top_percentile_selector.select.assert_called_once_with(population)
        
        assert self.network_crosser.cross.call_count == 3
        
        first_call_args = self.network_crosser.cross.call_args_list[0][0]
        second_call_args = self.network_crosser.cross.call_args_list[1][0]
        third_call_args = self.network_crosser.cross.call_args_list[2][0]
        
        assert first_call_args[0] == population.population[0]
        assert first_call_args[1] == top_network_1 or first_call_args[1] == top_network_2
        
        assert second_call_args[0] == population.population[1]
        assert second_call_args[1] == top_network_1 or second_call_args[1] == top_network_2
        
        assert third_call_args[0] == population.population[2]
        assert third_call_args[1] == top_network_1 or third_call_args[1] == top_network_2