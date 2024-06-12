from decimal import Decimal
from random import Random
from config import Config
from src.neuralnet.network import Network
from src.pipeline.population import PopulationDTO


class PopulationMutator():
    # Mutation percent is the chance that a node or synapse will be mutated
    # Mutation step is the scale that a node or synapse will be mutated by. For example, 0.1 would add or subtract a maximum of 0.1 from the node or synapse
    def __init__(self, mutation_percent: Decimal = Decimal(0.1), mutation_step: Decimal = Decimal(0.1)):
        self.mutation_percent: Decimal = mutation_percent
        self.mutation_step: Decimal = mutation_step
    
    def mutate(self, population: PopulationDTO) -> PopulationDTO:
        print(f'Mutating population of {len(population.population)} individuals.')
        for network in population.population:
            self.mutate_network(network)
            
        return population
            
    def mutate_network(self, network: Network):
        for layer in network.node_layers:
            for node in layer:
                if self.should_mutate():
                    node.bias += self.get_mutation_value()
        
        for layer in network.synapse_layers:
            for synapse in layer:
                if self.should_mutate():
                    synapse.weight += self.get_mutation_value()
                    
        network.learning_rate += self.get_mutation_value()

        return network
    
    def should_mutate(self) -> bool:
        return Random().random() < self.mutation_percent
    
    def get_mutation_value(self) -> Decimal:
        random_between_negative_one_and_one = (Random().random() * 2) - 1
        unquantized_value =  Decimal(random_between_negative_one_and_one) * Decimal(self.mutation_step)
        
        precision = Config().decimal_precision
        
        return unquantized_value.quantize(Decimal(10) ** -precision)