from decimal import Decimal
import unittest

from src.neuralnet.sigmoid_node import SigmoidNode
from src.neuralnet.sigmoid_node_to_dict import SigmoidNodeToDict
from src.neuralnet.synapse import Synapse


class TestSigmoidNodeToDict(unittest.TestCase):
    def test_to_dict_returns_training_state(self):
        sigmoid_node = SigmoidNode()
        sigmoid_node.id = 'test_id'
        sigmoid_node.bias = Decimal(0.5)
        
        sigmoid_node.activation = Decimal(0.2)
        sigmoid_node.loss = Decimal(0.1)
        sigmoid_node.gradients = [Decimal(0.1), Decimal(0.2)]
        
        dict_representation = SigmoidNodeToDict().to_dict(sigmoid_node)
        
        assert dict_representation['id'] == 'test_id' # type: ignore
        assert dict_representation['bias'] == Decimal(0.5) # type: ignore
        
        # should have no representation for the other fields.
        assert 'activation' not in dict_representation
        assert 'loss' not in dict_representation
        assert 'gradients' not in dict_representation