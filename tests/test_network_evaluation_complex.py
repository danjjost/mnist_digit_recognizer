import math
import unittest
from src.network import Network

from tests.test_networks.complex_network_creator import ComplexNetworkCreator

class TestNetworkEvaluationComplex(unittest.TestCase):
    
    def test_network_seven_node_evaluation(self):
        network = ComplexNetworkCreator().create_complex_network()
                
        
        network.evaluate()
        
        
        self.validate_layer_1_activation(network)
        self.validate_layer_2_activation(network)
        self.validate_layer_3_activation(network)
        
        
    def validate_layer_1_activation(self, network: Network):
        assert math.isclose(network.node_layers[0][0].activation, 0.9706, abs_tol=0.001)
        assert math.isclose(network.node_layers[0][1].activation, 0.9836, abs_tol=0.001)
        
        
    def validate_layer_2_activation(self, network: Network):
        assert math.isclose(network.node_layers[1][0].activation, 0.9642, abs_tol=0.001)
        assert math.isclose(network.node_layers[1][1].activation, 0.8784, abs_tol=0.001)
        assert math.isclose(network.node_layers[1][2].activation, 0.9407, abs_tol=0.001)
    
    
    def validate_layer_3_activation(self, network: Network):
        assert math.isclose(network.node_layers[2][0].activation, 0.9586, abs_tol=0.001)
        assert math.isclose(network.node_layers[2][1].activation, 0.9715, abs_tol=0.001)