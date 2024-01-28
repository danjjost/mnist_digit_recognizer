from typing import List

class Node:
    id: str = None
    
    # evaluation state
    starting_input: float = None

    activation: float = None

    # predefined state
    bias: float = 0

    input_synapses: List['Synapse'] = None
    output_synapses: List['Synapse'] = None
