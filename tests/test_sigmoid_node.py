import math
import unittest

from src.sigmoid_node import SigmoidNode


class TestSigmoidNode(unittest.TestCase):

    # Test With Two Input Nodes
    def test_constructor_generates_id(self):
        node = SigmoidNode()
        assert node.id != None
        assert node.id != ""

    # Test first layer
    def test_first_layer_throws_without_starting_inputs_or_synapses(self):
        node = SigmoidNode()
        node.starting_input = None
        node.input_synapses = None

        with self.assertRaises(ValueError):
            node.determine_activation()


    # Test clear evaluation state
    def test_clear_evaluation_state(self):
        node = SigmoidNode()
        
        node.activation = 1
        node.starting_input = 1
        
        node.clear_evaluation_state()

        assert node.activation == None
        assert node.starting_input == None
