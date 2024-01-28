import uuid
import math

from src.node import Node

class SigmoidNode(Node):
    def __init__(self) -> None:
        self.id = str(uuid.uuid4())


    def determine_activation(self) -> None:
        self.validate()

        net_input = self.get_net_input()

        self.activation = self.activation_function(net_input + self.bias)
        
        return self.activation


    def validate(self):
        if self.input_synapses is None and self.starting_input is None:
            raise ValueError(f"Node '{self.id}' appears to be a first-layer node, but has no starting input! If this is a first-layer node, please explicitly set the starting input.")

    def get_net_input(self):
        net_input = 0
        
        for synapse in self.input_synapses:
            net_input += synapse.input_node.activation * synapse.weight

        return net_input


    def activation_function(self, netInput: float) -> float:
        return 1 / (1 + math.exp(-netInput))
    

    def clear_evaluation_state(self) -> None:
        self.activation = None
        self.starting_input = None