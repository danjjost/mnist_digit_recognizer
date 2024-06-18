from typing import TYPE_CHECKING
import uuid

if TYPE_CHECKING:
    from src.neuralnet.sigmoid_node import SigmoidNode

class Synapse:
    def __init__(self, input_node: 'SigmoidNode', output_node: 'SigmoidNode', weight: float):
        input_node.output_synapses.append(self)
        
        output_node.input_synapses.append(self)

        self.id = str(uuid.uuid4())
        
        # evaluation state
        self.gradients: list[float] = []
        
        # predefined state
        self.input_node: SigmoidNode = input_node
        self.output_node: SigmoidNode = output_node
        self.weight:float = weight or 0.0
        
        
    def apply_gradients(self, learning_rate: float):
        self.weight += learning_rate * sum(self.gradients, 0.0)
        self.gradients.clear()
        
        
    def clear_evaluation_state(self):
        self.gradients.clear()