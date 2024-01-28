

from src.sigmoid_node import SigmoidNode


class Layer():
    nodes = []

    def __init__(self, size):
        self.nodes = []
        
        for i in range(size):
            self.nodes.append(SigmoidNode())