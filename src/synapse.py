from src.node import Node


class Synapse:
    
    def __init__(self, input_node: Node = None, output_node: Node = None, weight: float = None):
        self.input_node = input_node
        self.output_node = output_node
        self.weight = weight