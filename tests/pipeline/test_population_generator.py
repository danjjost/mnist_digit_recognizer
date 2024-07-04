
import unittest

from src.pipeline.population import PopulationDTO
from src.pipeline.population_modifiers.population_generator import PopulationGenerator


class TestPopulationGenerator(unittest.TestCase):
    def test_population_generator(self):
        count = 100
        schema: list[int] = [5, 5, 5]
                
        population_generator = PopulationGenerator()
        
        
        population = population_generator.generate(count, schema)
        
        self.verify_values_dont_match(population, count, schema)
        self.verify_valid_values(population, count, schema)
        
    def verify_values_dont_match(self, populationDTO: PopulationDTO, count: int, schema: list[int]):
        # all node biases should be different
        biases: set[float] = set()
        for network in populationDTO.population:
            for node_layer in network.node_layers:
                for node in node_layer:
                    biases.add(node.bias)
            
        synapse_weights: set[float] = set()
        for network in populationDTO.population:
            for synapse_layer in network.synapse_layers:
                for synapse in synapse_layer:
                    synapse_weights.add(synapse.weight)

        assert len(biases) > 1
        assert len(synapse_weights) > 1
        
    def verify_valid_values(self, populationDTO: PopulationDTO, count: int, schema: list[int]):
        assert len(populationDTO.population) == count

    
        for network in populationDTO.population:        
            for index, num_nodes in enumerate(schema):
                assert len(network.node_layers[index]) == num_nodes
                for node in network.node_layers[index]:
                    assert node.bias is not None
                    assert node.bias >= 0 and node.bias <= 1
                    if node.output_synapses is not None and len(node.output_synapses) > 0: # type: ignore
                        for synapse in node.output_synapses:
                            assert synapse.weight >= 0 and synapse.weight <= 1
                    