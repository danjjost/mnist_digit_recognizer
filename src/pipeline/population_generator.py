from decimal import Decimal
import random
from src.neuralnet.network import Network
from src.pipeline.population import PopulationDTO
from src.pipeline.population_modifier import PopulationModifier


class PopulationGenerator(PopulationModifier):
    def generate(self, count: int, schema: list[int]) -> PopulationDTO:
        networks: list[Network] = []
        
        for _ in range(count):
            network = Network(schema)
            self.randomize(network)
            networks.append(network)
            
        return PopulationDTO(networks)
            
            
    def randomize(self, network: Network):
        for node_layer in network.node_layers:
            for node in node_layer:
                node.bias = Decimal(random.random())
                
        for synapse_layer in network.synapse_layers:
            for synapse in synapse_layer:
                synapse.weight = Decimal(random.random())
                
        network.learning_rate = Decimal(random.random())
        
        return network

        