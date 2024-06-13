import math
import uuid

from src.neuralnet.synapse import Synapse

class SigmoidNode():
    def __init__(self) -> None:        
        self.id = str(uuid.uuid4())
        
        # evaluation state
        self.starting_input: float | None = None
        self.activation: float = float(0)
        self.loss: float = float(0)
        self.gradients: list[float] = []
        
        # predefined state
        self.bias: float = float('0')

        self.input_synapses: list[Synapse] = []
        self.output_synapses: list[Synapse] = []
        
    def apply_gradients(self, learning_rate: float):
        self.bias += learning_rate * sum(self.gradients)
        self.gradients.clear()

    def determine_activation(self) -> float:
        self.validate()

        net_input = self.get_net_input()

        self.activation = self.activation_function(net_input + self.bias)
        
        return self.activation


    def validate(self):
        if self.starting_input is not None and (len(self.input_synapses) > 0):
            raise ValueError(f"Node '{self.id}' appears to be a first-layer node, but has input synapses and starting input! If this is a first-layer node, please remove the input synapses.")
        
        
        if (len(self.input_synapses) == 0) and self.starting_input is None:
            raise ValueError(f"Node '{self.id}' appears to be a first-layer node, but has no starting input! If this is a first-layer node, please explicitly set the starting input.")


    def get_net_input(self):
        net_input = float('0')
        
        if self.starting_input is not None:
            return self.starting_input
        
        for synapse in self.input_synapses:
            net_input += synapse.input_node.activation * synapse.weight

        return net_input


    def activation_function(self, netInput: float) -> float:
        return 1 / (1 + math.exp(-netInput))
    

    def clear_evaluation_state(self) -> None:
        self.starting_input = None
        self.activation = float(0)
        self.loss = float(0)
        
    def to_dict(self):
        return {
            'id': self.id,
            'bias': self.bias,
        }
        