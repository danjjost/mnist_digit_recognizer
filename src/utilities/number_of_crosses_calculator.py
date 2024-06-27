from config import Config
from src.neuralnet.network import Network


class NumberOfCrossesCalculator:
    def __init__(self, config: Config):
        self.config = config
    
    def get_number_of_crosses(self, network: Network) -> int:
        number_of_possible_crosses = self.get_number_of_possible_crosses(network)
        
        number_of_crosses = int(number_of_possible_crosses * self.config.cross_percent)
        
        if number_of_crosses == 0:
            number_of_crosses = 1
        
        return number_of_crosses
    
    
    def get_number_of_possible_crosses(self, network: Network) -> int:
        number_of_synapses = self.get_number_of_synapses(network)
        number_of_nodes = self.get_number_of_nodes(network)
        
        return number_of_nodes + number_of_synapses
    
    
    def get_number_of_synapses(self, network: Network) -> int:
        number_of_synapses = 0
        
        for layer in network.synapse_layers:
            number_of_synapses += len(layer)
        
        return number_of_synapses
    
    
    def get_number_of_nodes(self, network: Network) -> int:
        number_of_nodes = 0
        
        for layer in network.node_layers:
            number_of_nodes += len(layer)
        
        return number_of_nodes
    