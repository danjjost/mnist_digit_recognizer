import unittest

from src.neuralnet.network import Network
from src.pipeline.competition import Competition
from src.pipeline.epoch import Epoch

class TestEpoch(unittest.TestCase):
    def test_epoch_runs_competition_for_each_network(self):
        population = [Network([1, 1]), Network([1, 1])]
        competition = MockCompetition()
        
        epoch = Epoch(competition)
        
        epoch.run(population)
        
        assert len(competition.competitions) == 4, f"Expected 4 competitions, got {len(competition.competitions)}"
        assert (population[0].id, population[0].id) in competition.competitions, f"Expected competition between {population[0].id} and {population[0].id}"
        assert (population[0].id, population[1].id) in competition.competitions, f"Expected competition between {population[0].id} and {population[1].id}"
        assert (population[1].id, population[0].id) in competition.competitions, f"Expected competition between {population[1].id} and {population[0].id}"
        assert (population[1].id, population[1].id) in competition.competitions, f"Expected competition between {population[1].id} and {population[1].id}"
        
        
    def test_competition_modification_is_applied_to_population(self):
        network_1 = Network([1, 1])
        network_2 = Network([1, 1])
        
        network_1.synapse_layers[0][0].weight = 0
        network_2.synapse_layers[0][0].weight = 0
        
        population = [network_1, network_2]
        
        competition = MockCompetition(override_weights=1)
        
        epoch = Epoch(competition)

        
        epoch.run(population)
        
        
        assert population[0].synapse_layers[0][0].weight == 1, f"Expected weight of 1, got {population[0].synapse_layers[0][0].weight}"
        assert population[1].synapse_layers[0][0].weight == 1, f"Expected weight of 1, got {population[1].synapse_layers[0][0].weight}"
        

        
class MockCompetition(Competition):
    def __init__(self, override_weights: float = None):
        self.competitions = []
        self.override_weights = override_weights
    
    def compete(self, challenger: Network, challenged: Network):
        self.competitions.append((challenger.id, challenged.id))
        if(self.override_weights != None):
            challenger.synapse_layers[0][0].weight = self.override_weights
            challenged.synapse_layers[0][0].weight = self.override_weights