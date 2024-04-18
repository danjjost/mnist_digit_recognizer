from decimal import Decimal, getcontext
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from src.synapse import Synapse

from src.synapse import Synapse

class Node:
    def __init__(self) -> None:
        self.id: str = None
        
        # evaluation state
        self.starting_input: Decimal = None
        self.activation: Decimal = None
        self.loss: Decimal = None
        self.gradients: List[Decimal] = []
        
        # predefined state
        self.bias: Decimal = Decimal('0')

        self.input_synapses: List['Synapse'] = []
        self.output_synapses: List['Synapse'] = []
        
    def apply_gradients(self, learning_rate: Decimal):
        self.bias += learning_rate * sum(self.gradients)