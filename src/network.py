from src.sigmoid_node import SigmoidNode
from src.synapse import Synapse


class Network():

    def __init__(self, dimensions):
        self.node_layers: list[list[SigmoidNode]] = []
        self.synapse_layers: list[list[Synapse]] = []
        
        self.initialize(dimensions)


    def initialize(self, dimensions):
        self.initialize_node_layers(dimensions)
        self.initialize_synapse_layers(dimensions)

    def initialize_node_layers(self, dimensions):
        for layer_index in range(len(dimensions)):
            self.node_layers.append(self.create_node_layer(dimensions[layer_index]))
            
    def create_node_layer(self, size):
        layer = []

        for node_index in range(size):
            layer.append(SigmoidNode())

        return layer
            
    def initialize_synapse_layers(self, network_dimensions):
        for synapse_layer_index in range(len(network_dimensions) - 1):
            self.synapse_layers.append([])
            
        for node_layer_index in range(len(network_dimensions) - 1):
            current_layer_index = node_layer_index
            next_layer_index = node_layer_index + 1
            self.synapse_layers[current_layer_index] = self.create_synapse_layer(current_layer_index, next_layer_index)
            
    def create_synapse_layer(self, current_layer_index, next_layer_index):
        synapse_layer = []
        
        for current_node in self.node_layers[current_layer_index]:
            for next_node in self.node_layers[next_layer_index]:
                synapse_layer.append(Synapse(current_node, next_node))
                
        return synapse_layer