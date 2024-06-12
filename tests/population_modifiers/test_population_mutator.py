from decimal import Decimal
import unittest

from src.neuralnet.network import Network
from src.pipeline.population import PopulationDTO
from src.pipeline.population_modifiers.population_mutator import PopulationMutator


class TestPopulationMutator(unittest.TestCase):
    def test_mutates_population(self):
        mutator = PopulationMutator(Decimal(1), Decimal(1))
        
        network1 = Network([1, 1])
        network1.node_layers[0][0].bias = Decimal(0.0)        
        network1.synapse_layers[0][0].weight = Decimal(0.0)
        
        network2 = Network([1, 1])
        network2.node_layers[0][0].bias = Decimal(0.0)
        network2.synapse_layers[0][0].weight = Decimal(0.0)
        
        population = PopulationDTO([network1, network2])
        
        mutator.mutate(population)
        
        assert network1.node_layers[0][0].bias != Decimal(0.0)
        assert network1.synapse_layers[0][0].weight != Decimal(0.0)
        
        assert network2.node_layers[0][0].bias != Decimal(0.0)
        assert network2.synapse_layers[0][0].weight != Decimal(0.0)
        
        assert network1.node_layers[0][0].bias > Decimal(-1.0)
        assert network1.node_layers[0][0].bias < Decimal(1.0)
        
        assert network1.synapse_layers[0][0].weight > Decimal(-1.0)
        assert network1.synapse_layers[0][0].weight < Decimal(1.0)
        
        assert network2.node_layers[0][0].bias > Decimal(-1.0)
        assert network2.node_layers[0][0].bias < Decimal(1.0)
        
        assert network2.synapse_layers[0][0].weight > Decimal(-1.0)
        assert network2.synapse_layers[0][0].weight < Decimal(1.0)
        
    def test_does_not_mutate_when_odds_are_zero(self):
        mutator = PopulationMutator(Decimal(0), Decimal(1))
        
        network1 = Network([1, 1])
        network1.node_layers[0][0].bias = Decimal(0.0)        
        network1.synapse_layers[0][0].weight = Decimal(0.0)
        
        network2 = Network([1, 1])
        network2.node_layers[0][0].bias = Decimal(0.0)
        network2.synapse_layers[0][0].weight = Decimal(0.0)
        
        population = PopulationDTO([network1, network2])
        
        mutator.mutate(population)
        
        assert network1.node_layers[0][0].bias == Decimal(0.0)
        assert network1.synapse_layers[0][0].weight == Decimal(0.0)
        
        assert network2.node_layers[0][0].bias == Decimal(0.0)
        assert network2.synapse_layers[0][0].weight == Decimal(0.0)
        
    def test_mutates_learning_rate(self):
        mutator = PopulationMutator(Decimal(1), Decimal(1))
        
        network1 = Network([1, 1])
        network1.learning_rate = Decimal(0.0)
                
        network2 = Network([1, 1])
        network2.learning_rate = Decimal(0.0)
        
        population = PopulationDTO([network1, network2])
        
        mutator.mutate(population)
        
        assert network1.learning_rate != Decimal(0.0)
        assert network2.learning_rate != Decimal(0.0)
        
        assert network1.learning_rate > Decimal(-1.0)
        assert network1.learning_rate < Decimal(1.0)
        
        