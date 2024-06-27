import unittest

from config import Config
from src.neuralnet.network import Network
from src.utilities.number_of_crosses_calculator import NumberOfCrossesCalculator


class TestNumberOfCrossesCalculator(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config()
        
        self.number_of_crosses_calcuator = NumberOfCrossesCalculator(self.config)
    
    def test_calculates_number_of_crosses(self):
        self.config.cross_percent = 0.3
        
        # 21 *0.3 = 6.3, flattens to 6
        expected_number_of_crosses = 6
        
        
        # 10 synapses, 11 biases = 21 total values
        network = Network([10, 1])
        
        
        number_of_crosses = self.number_of_crosses_calcuator.get_number_of_crosses(network)
        
        
        assert number_of_crosses == expected_number_of_crosses, f"Expected: {expected_number_of_crosses}, got: {number_of_crosses}"
        
    def test_calculates_number_of_crosses_when_less_than_one(self):
        self.config.cross_percent = 0.1
        
        # 3 * 0.1 = 0.3. We always round zero up to 1.
        expected_number_of_crosses = 1
        
        
        # 1 synapse, 2 biases = 3 total values
        network = Network([1, 1])
        
        
        number_of_crosses = self.number_of_crosses_calcuator.get_number_of_crosses(network)
        
        
        assert number_of_crosses == expected_number_of_crosses