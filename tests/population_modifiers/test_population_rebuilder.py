from decimal import Decimal
from src.neuralnet.network import Network
from src.pipeline.population import PopulationDTO
from src.pipeline.population_modifiers.population_rebuilder import PopulationRebuilder


class TestPopulationRebuilder():
    def test_clones_random_members_of_population(self):
        network1 = Network([1])
        network1.node_layers[0][0].bias = Decimal(0.1)
        
        network2 = Network([1])
        network1.node_layers[0][0].bias = Decimal(0.2)
        
        network3 = Network([1])
        network1.node_layers[0][0].bias = Decimal(0.3)
        
        population = PopulationDTO([network1, network2, network3])
        
        PopulationRebuilder().rebuild(population=population, number_to_copy=2)

        assert len(population.population) == 5
        assert network1.id in [network.id for network in population.population]
        assert network2.id in [network.id for network in population.population]
        assert network3.id in [network.id for network in population.population]

        all_biases = [network.node_layers[0][0].bias for network in population.population]
        for network in population.population:
            assert network.node_layers[0][0].bias in all_biases