from decimal import Decimal

from typing import TYPE_CHECKING
import uuid

if TYPE_CHECKING:
    from src.neuralnet.sigmoid_node import SigmoidNode

class Synapse:
    def __init__(self, input_node: 'SigmoidNode', output_node: 'SigmoidNode', weight: Decimal):
        input_node.output_synapses.append(self)
        
        output_node.input_synapses.append(self)

        self.id = str(uuid.uuid4())
        
        # evaluation state
        self.gradients: list[Decimal] = []
        
        # predefined state
        self.input_node: SigmoidNode = input_node
        self.output_node: SigmoidNode = output_node
        self.weight = weight or Decimal(0)
        
        
    def apply_gradients(self, learning_rate: Decimal):
        self.weight += learning_rate * sum(self.gradients, Decimal('0'))
        self.gradients.clear()
        
        
    def clear_evaluation_state(self):
        self.gradients = []