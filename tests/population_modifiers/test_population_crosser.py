import unittest
from unittest.mock import MagicMock

from config import Config
from src.neuralnet.network import Network
from src.neuralnet.sigmoid_node import SigmoidNode
from src.neuralnet.synapse import Synapse
from src.pipeline.network_crosser import NetworkCrosser
from src.pipeline.population_generator import PopulationGenerator
from src.pipeline.population_modifiers.population_crosser import PopulationCrosser
from src.utilities.number_of_crosses_calculator import NumberOfCrossesCalculator


class TestPopulationCrosser(unittest.TestCase):
    def test_cross_swaps_at_least_one_gene_per_network(self):
        population = PopulationGenerator().generate(2, [1, 1])
        config = Config()
        config.cross_percent = 0.01 # 1% rounds up to 1 gene
        population_crosser = PopulationCrosser(config)
        
        output_population = population_crosser.run(population)
        
        assert self.has_one_matching_synapse(output_population.population[0], output_population.population[1]) or self.has_one_matching_bias(output_population.population[0], output_population.population[1])
    
    def setUp(self) -> None:
        self.config = Config()

        self.default_number_of_crosses = 1

        self.network_crosser = MagicMock(spec=NetworkCrosser)
        self.number_of_crosses_calculator = MagicMock(spec=NumberOfCrossesCalculator)
        self.number_of_crosses_calculator.get_number_of_crosses.return_value = self.default_number_of_crosses
        
        
    def test_calls_network_crosser_for_each_network(self):
        population = PopulationGenerator().generate(3, [1, 1])
        
        population_crosser = PopulationCrosser(Config(), self.network_crosser, self.number_of_crosses_calculator)
    
    
        population_crosser.run(population)
    
    
        assert self.network_crosser.cross.call_count == 3
        assert self.network_crosser.cross.call_args_list[0][0][0] == population.population[0]
        assert self.network_crosser.cross.call_args_list[1][0][0] == population.population[1]
        assert self.network_crosser.cross.call_args_list[2][0][0] == population.population[2]

    # Helper methods
    
    def has_one_matching_synapse(self, network1: Network, network2: Network) -> bool:
        network1_synapses = self.list_all_synapses(network1)            
        network2_synapses = self.list_all_synapses(network2)
        
        for synapse_index in range(len(network1_synapses)):
            if network1_synapses[synapse_index] == network2_synapses[synapse_index]:
                return True
            
        return False
    
    def has_one_matching_bias(self, network1: Network, network2: Network) -> bool:
        network1_biases = self.list_all_nodes(network1)
        network2_biases = self.list_all_nodes(network2)
        
        for bias_index in range(len(network1_biases)):
            if network1_biases[bias_index] == network2_biases[bias_index]:
                return True
            
        return False
    
    def list_all_nodes(self, network: Network) -> list[SigmoidNode]:
        nodes: list[SigmoidNode] = []
        
        for layer in network.node_layers:
            for node in layer:
                nodes.append(node)
                
        return nodes
    
    def list_all_synapses(self, network: Network) -> list[Synapse]:
        synapses: list[Synapse] = []
        
        for layer in network.synapse_layers:
            for synapse in layer:
                synapses.append(synapse)
                
        return synapses