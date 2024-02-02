import math
import unittest
from src.node import Node

from src.sigmoid_node import SigmoidNode
from src.synapse import Synapse


class TestSigmoidNodeSimpleExample(unittest.TestCase):

    def test_two_input_nodes(self):
        node = SigmoidNode()
        node.bias = 2
        
        node.input_synapses = [
            self.createSynapse(activation=1, weight=3),
            self.createSynapse(activation=0.5, weight=2)
        ]
        
        node.determine_activation()

        expected_activation = self.get_expected_activation()

        assert math.isclose(node.activation, expected_activation, abs_tol=0.001)


    def get_expected_activation(self):
        netInput = (1*3) + (0.5*2)

        netInputWithBias = netInput + 2
        
        return self.sigmoid(netInputWithBias)
    
    
    def createSynapse(self, activation, weight):
        synapse = Synapse()
        synapse.weight = weight
        synapse.input_node = Node()
        synapse.input_node.activation = activation
        return synapse


    def sigmoid(self, net_input: float):
        return 1 / (1 + math.exp(-net_input))