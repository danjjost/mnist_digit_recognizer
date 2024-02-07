from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from src.synapse import Synapse

from src.synapse import Synapse

class Node:
    def __init__(self) -> None:
        self.id: str = None
        
        # evaluation state
        self.starting_input: float = None
        self.activation: float = None
        self.loss: float = None

        # predefined state
        self.bias: float = 0

        self.input_synapses: List['Synapse'] = []
        self.output_synapses: List['Synapse'] = []