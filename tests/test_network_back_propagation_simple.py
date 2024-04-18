import math
import unittest
from tests.test_networks.simple_network_creator import SimpleNetworkCreator


class TestNetworkBackPropagation(unittest.TestCase):
    def test_back_propagation(self):
        network = SimpleNetworkCreator().create_simple_network()
        
        network.feed_forward()
        
        targetOutputs = [0.25]
        
        network.back_propagate(targetOutputs)
        network.apply_gradients()
        
        assert math.isclose(network.node_layers[1][0].bias, 0.92166, abs_tol=0.05)
        assert math.isclose(network.node_layers[0][0].bias, 1.993294, abs_tol=0.05)
        assert math.isclose(network.node_layers[0][1].bias, 2.999495, abs_tol=0.05)
        
        assert math.isclose(network.synapse_layers[0][0].weight, 0.22396, abs_tol=0.05)
        assert math.isclose(network.synapse_layers[0][1].weight, 0.32294, abs_tol=0.05)
        
    def test_back_propagation_multiple_gradients(self):
        network = SimpleNetworkCreator().create_simple_network()
        
        network.feed_forward()
        
        targetOutputs = [0.25]
        
        # this puts two gradients on each synapse and node
        network.back_propagate(targetOutputs)
        network.back_propagate(targetOutputs)
        
        
        network.apply_gradients()
        
        
        assert math.isclose(network.node_layers[1][0].bias, 0.84332, abs_tol=0.05)
        assert math.isclose(network.node_layers[0][0].bias, 1.986588, abs_tol=0.05)
        assert math.isclose(network.node_layers[0][1].bias, 2.99899, abs_tol=0.05)
        
        assert math.isclose(network.synapse_layers[0][0].weight, 0.14792, abs_tol=0.05)
        assert math.isclose(network.synapse_layers[0][1].weight, 0.24588, abs_tol=0.05)

        
        