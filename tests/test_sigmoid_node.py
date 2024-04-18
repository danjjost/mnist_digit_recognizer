from decimal import Decimal
import math
import unittest

from src.sigmoid_node import SigmoidNode
from src.synapse import Synapse


class TestSigmoidNode(unittest.TestCase):

    def test_constructor_generates_id(self):
        node = SigmoidNode()
        assert node.id != None
        assert node.id != ""


    def test_first_layer_throws_without_starting_input_or_synapses(self):
        node = SigmoidNode()
        
        node.starting_input = None
        node.input_synapses = None

        with self.assertRaises(ValueError):
            node.determine_activation()


    def test_first_layer_throws_if_starting_input_and_synapses_are_present(self):
        node = SigmoidNode()
        
        node.starting_input = 1
        node.input_synapses = [Synapse()]

        with self.assertRaises(ValueError):
            node.determine_activation()


    def test_first_layer_uses_starting_input_if_no_input_synapses_are_present(self):
        node = SigmoidNode()
        
        node.starting_input = 0.4
        node.bias = 1
        
        node.input_synapses = None
        
        node.determine_activation()
        
        assert math.isclose(node.activation, self.sigmoid(node.starting_input + node.bias), abs_tol=0.001)

    def test_clear_evaluation_state(self):
        node = SigmoidNode()
        
        node.activation = 1
        node.starting_input = 1
        node.loss = 1
        
        node.clear_evaluation_state()

        assert node.activation == None
        assert node.starting_input == None
        assert node.loss == None


    def test_throws_if_previous_node_is_not_active(self):
        node = SigmoidNode()
        
        previous_node = SigmoidNode()
        
        synapse = Synapse(previous_node, node)
        
        node.input_synapses = [synapse]
        previous_node.activation = None

        with self.assertRaises(ValueError):
            node.determine_activation()


    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))