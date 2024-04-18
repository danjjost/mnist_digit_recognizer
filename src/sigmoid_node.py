from decimal import Decimal
import uuid
import math

from src.node import Node

class SigmoidNode(Node):
    def __init__(self) -> None:
        super().__init__()
        self.id = str(uuid.uuid4())


    def determine_activation(self) -> None:
        self.validate()

        net_input = self.get_net_input()

        self.activation = self.activation_function(net_input + self.bias)
        
        return self.activation


    def validate(self):
        if self.starting_input is not None and (self.input_synapses is not None and len(self.input_synapses) > 0):
            raise ValueError(f"Node '{self.id}' appears to be a first-layer node, but has input synapses and starting input! If this is a first-layer node, please remove the input synapses.")
        
        
        if (self.input_synapses is None or len(self.input_synapses) == 0) and self.starting_input is None:
            raise ValueError(f"Node '{self.id}' appears to be a first-layer node, but has no starting input! If this is a first-layer node, please explicitly set the starting input.")

    def get_net_input(self):
        net_input = Decimal('0')
        
        if self.starting_input is not None:
            return self.starting_input
        
        for synapse in self.input_synapses:
            if synapse.input_node.activation is None:
                raise ValueError(f"Node '{self.id}' has an input synapse from node '{synapse.input_node.id}' that has not been activated!")

            net_input += synapse.input_node.activation * synapse.weight

        return net_input


    def activation_function(self, netInput: Decimal) -> Decimal:
        return Decimal(1) / (Decimal(1) + Decimal(-netInput).exp())
    

    def clear_evaluation_state(self) -> None:
        self.activation = None
        self.starting_input = None
        self.loss = None
        self.gradients = []