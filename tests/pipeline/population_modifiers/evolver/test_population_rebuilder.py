
import unittest
from unittest.mock import MagicMock
from src.neuralnet.network import Network
from src.pipeline.population import PopulationDTO
from src.pipeline.population_modifiers.evolver.population_rebuilder import PopulationRebuilder
from src.pipeline.population_modifiers.population_generator import PopulationGenerator
from src.utilities.top_percentile_selector import TopPercentileSelector


class TestPopulationRebuilder(unittest.TestCase):
    def test_calls_top_percentile_selector_with_input_population(self):
        population = PopulationGenerator().generate(4, [1,1])
        
        top_percentile_selector = MagicMock(spec=TopPercentileSelector)
        population_rebuilder = PopulationRebuilder(top_percentile_selector)
        
        # Act
        population_rebuilder.rebuild(population, 2)
        
        # Assert
        assert top_percentile_selector.select.called_with(population)
        
    def test_clones_networks_from_top_percentile(self):
        # Arrange
        full_population = PopulationGenerator().generate(10, [1,1])
        selected_population = full_population.population[:2]
        
        top_percentile_selector = MagicMock(spec=TopPercentileSelector)
        top_percentile_selector.select.return_value = selected_population
        
        population_rebuilder = PopulationRebuilder(top_percentile_selector)


        # Act
        rebuilt_population = population_rebuilder.rebuild(full_population, 2)
        

        # Assert
        assert len(full_population.population) == 12
        assert self.has_clone(selected_population[0], rebuilt_population)
        assert self.has_clone(selected_population[1], rebuilt_population)
        
        
    def test_clones_subset_of_top_percentile(self):
        # Arrange
        full_population = PopulationGenerator().generate(10, [1,1])
        selected_population = full_population.population[:2]
        
        top_percentile_selector = MagicMock(spec=TopPercentileSelector)
        top_percentile_selector.select.return_value = selected_population
        
        population_rebuilder = PopulationRebuilder(top_percentile_selector)


        # Act
        rebuilt_population = population_rebuilder.rebuild(full_population, 1)
        

        # Assert
        assert len(full_population.population) == 11
        assert self.has_clone(selected_population[0], rebuilt_population) or self.has_clone(selected_population[1], rebuilt_population)
        
        

    # Helpers
    def has_clone(self, network: Network, population: PopulationDTO):
        for individual in population.population:
            if self.all_weights_equal(individual, network) and self.all_biases_equal(individual, network):
                return True
            
        return False
    
    def all_weights_equal(self, network1: Network, network2: Network):
        for layer_index in range(len(network1.synapse_layers)):
            for synapse_index in range(len(network1.synapse_layers[layer_index])):
                if network1.synapse_layers[layer_index][synapse_index].weight != network2.synapse_layers[layer_index][synapse_index].weight:
                    return False
                
        return True
    
    def all_biases_equal(self, network1: Network, network2: Network):
        for layer_index in range(len(network1.node_layers)):
            for node_index in range(len(network1.node_layers[layer_index])):
                if network1.node_layers[layer_index][node_index].bias != network2.node_layers[layer_index][node_index].bias:
                    return False
        return True