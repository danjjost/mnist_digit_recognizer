from decimal import Decimal
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from src.node import Node

class Synapse:
    def __init__(self, input_node: 'Node' = None, output_node: 'Node' = None, weight: Decimal = None):
        if input_node is not None:
            input_node.output_synapses.append(self)
        
        if output_node is not None:
            output_node.input_synapses.append(self)
        
        # evaluation state
        self.gradients: List[Decimal] = []
        
        # predefined state
        self.input_node = input_node
        self.output_node = output_node
        self.weight = weight
        
    def apply_gradients(self, learning_rate: Decimal):
        self.weight += learning_rate * sum(self.gradients, Decimal('0'))
        
    def clear_evaluation_state(self):
        self.gradients = []