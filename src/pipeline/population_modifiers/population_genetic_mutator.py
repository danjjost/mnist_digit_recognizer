from random import Random
from config import Config
from src.neuralnet.network import Network
from src.pipeline.population import PopulationDTO
from src.pipeline.population_modifier import PopulationModifier


class PopulationGeneticMutator(PopulationModifier):
    def __init__(self, config: Config):
        self.mutation_percentage = config.percent_mutation
        
    def run(self, population: PopulationDTO) -> PopulationDTO:
        print("Genetic Mutator - Running...")
        number_of_possible_mutations = self.get_number_of_possible_mutations(population)
        
        number_of_mutations = int(number_of_possible_mutations * self.mutation_percentage)
        if number_of_mutations == 0:
            number_of_mutations = 1
        
        for network in population.population:
            for _ in range(number_of_mutations):
                self.mutate(network, population)

        return population
    
    def mutate(self, network: Network, population: PopulationDTO):
        if self.get_random_boolean():
            self.mutate_synapse(network, population)
        else:
            self.mutate_node(network, population)
        
    def get_random_boolean(self) -> bool:
        return Random().random() > 0.5
    
    def mutate_synapse(self, network: Network, population: PopulationDTO):
        random_donor_network = Random().choice(population.population)
        
        random_synapse_layer_index = Random().randint(0, len(network.synapse_layers) - 1)
        random_synapse_index = Random().randint(0, len(network.synapse_layers[random_synapse_layer_index]) - 1)
        
        network.synapse_layers[random_synapse_layer_index][random_synapse_index] = random_donor_network.synapse_layers[random_synapse_layer_index][random_synapse_index]
        
    def mutate_node(self, network: Network, population: PopulationDTO):
        random_donor_network = Random().choice(population.population)
        
        random_node_layer_index = Random().randint(0, len(network.node_layers) - 1)
        random_node_index = Random().randint(0, len(network.node_layers[random_node_layer_index]) - 1)
        
        network.node_layers[random_node_layer_index][random_node_index] = random_donor_network.node_layers[random_node_layer_index][random_node_index]    
    
    def get_number_of_possible_mutations(self, population: PopulationDTO) -> int:
        number_of_synapses = self.get_number_of_synapses(population)
        number_of_nodes = self.get_number_of_nodes(population)
        
        return number_of_nodes + number_of_synapses
    
    def get_number_of_synapses(self, population: PopulationDTO) -> int:
        first_network = population.population[0]
        
        number_of_synapses = 0
        
        for layer in first_network.synapse_layers:
            number_of_synapses += len(layer)
        
        return number_of_synapses
    
    
    def get_number_of_nodes(self, population: PopulationDTO) -> int:
        first_network = population.population[0]
        
        number_of_nodes = 0
        
        for layer in first_network.node_layers:
            number_of_nodes += len(layer)
        
        return number_of_nodes
    