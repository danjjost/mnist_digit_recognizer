from decimal import Decimal
import unittest

from src.neuralnet.network import Network
from src.pipeline.population_generator import PopulationGenerator
from tests.neuralnet.weather_predictor.weather_conditions import WeatherConditions
from tests.neuralnet.weather_predictor.weather_predictor_data_generator import WeatherPredictorDataGenerator


class TestWeatherPredictionNetwork(unittest.TestCase):
    def test_training_on_single_data_point_inches_closer_to_correctness(self):
        network = Network([2, 3, 1])
        self.blank(network)
        
        
        temperature = Decimal(50)
        cloud_cover = Decimal(0.7)
        
        
        network.set_input([temperature, cloud_cover])
        network.feed_forward()
        
        
        original_output = network.get_outputs()[0]
        
        
        network.back_propagate([Decimal(1)])
        
        network.apply_gradients()
        
        network.set_input([temperature, cloud_cover])
        network.feed_forward()
        
        
        back_propagated_output = network.get_outputs()[0]
        

        assert back_propagated_output > original_output
    
    def test_back_propagation_can_solve_the_problem(self):
        network = Network([2, 1])
        network.learning_rate = Decimal(0.1)
        PopulationGenerator().randomize(network)
        
        rainy_test_cases = WeatherPredictorDataGenerator().generate_rainy_data(5000)
        sunny_test_cases = WeatherPredictorDataGenerator().generate_sunny_data(5000)
         
        for i in range(3000):
            if i > 1000:
                network.learning_rate = Decimal(0.01)
                if i > 1500:
                    network.learning_rate = Decimal(0.001)
                    if i > 2000:
                        network.learning_rate = Decimal(0.0001)
                        if i > 3000:
                            network.learning_rate = Decimal(0.00001)
                            if i > 4000: 
                                network.learning_rate = Decimal(0.000001)
            
            self.train_rainy_case(rainy_test_cases[i], network)
            self.train_sunny_case(sunny_test_cases[i], network)
            
                        
        rainy_test_cases = WeatherPredictorDataGenerator().generate_rainy_data(100)
        sunny_test_cases = WeatherPredictorDataGenerator().generate_sunny_data(100)
        
        rainy_correct = 0
        sunny_correct = 0
        
        for i in range(100):
            network.set_input([rainy_test_cases[i].temperature, rainy_test_cases[i].cloud_cover])
            network.feed_forward()
            
            if network.get_outputs()[0] > Decimal(0.5):
                rainy_correct += 1
                
            network.set_input([sunny_test_cases[i].temperature, sunny_test_cases[i].cloud_cover])
            network.feed_forward()
            
            if network.get_outputs()[0] < Decimal(0.5):
                sunny_correct += 1
                
        assert rainy_correct > 70, f"Rainy correct: {rainy_correct}"
        assert sunny_correct > 70, f"Sunny correct: {sunny_correct}"
            

    def train_rainy_case(self, rainy_case: WeatherConditions, network: Network):            
        network.set_input([rainy_case.temperature, rainy_case.cloud_cover])

        network.feed_forward()

        network.back_propagate([Decimal(1)])

        network.apply_gradients()
        
    
    def train_sunny_case(self, sunny_case: WeatherConditions, network: Network):
        network.set_input([sunny_case.temperature, sunny_case.cloud_cover])

        network.feed_forward()

        network.back_propagate([Decimal(0)])

        network.apply_gradients()
        
        
    def blank(self, network: Network):
        for layer in network.node_layers:
            for node in layer:
                node.bias = Decimal(0)

        for layer in network.synapse_layers:
            for synapse in layer:
                synapse.weight = Decimal(0)