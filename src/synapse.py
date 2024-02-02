from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.node import Node

class Synapse:
    def __init__(self, input_node: 'Node' = None, output_node: 'Node' = None, weight: float = None):
        if input_node is not None:
            input_node.output_synapses.append(self)
        
        if output_node is not None:
            output_node.input_synapses.append(self)
        
        self.input_node = input_node
        self.output_node = output_node
        self.weight = weight