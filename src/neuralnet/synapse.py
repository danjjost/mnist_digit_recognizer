from decimal import Decimal

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.neuralnet.sigmoid_node import SigmoidNode

class Synapse:
    def __init__(self, input_node: 'SigmoidNode', output_node: 'SigmoidNode', weight: Decimal):
        input_node.output_synapses.append(self)
        
        output_node.input_synapses.append(self)
        
        # evaluation state
        self.gradients: list[Decimal] = []
        
        # predefined state
        self.input_node: SigmoidNode = input_node
        self.output_node: SigmoidNode = output_node
        self.weight = weight or 0
        
    def apply_gradients(self, learning_rate: Decimal):
        self.weight += learning_rate * sum(self.gradients, Decimal('0'))
        
    def clear_evaluation_state(self):
        self.gradients = []