
from typing import Union
import unittest
from unittest.mock import Mock

from src.neuralnet.network import Network
from src.pipeline.population_modifiers.epoch.competition import Competition
from src.pipeline.population_modifiers.epoch.competition_epoch import CompetitionEpoch
from src.pipeline.population import PopulationDTO

class TestCompetitionEpoch(unittest.TestCase):
    def test_epoch_runs_competition_for_each_network(self):
        population = [Network([1, 1]), Network([1, 1])]
        populationDto = PopulationDTO(population)
        
        competition = Mock(spec=Competition)
        
        epoch = CompetitionEpoch(competition)
        
        epoch.run(populationDto)

        competition.compete.assert_any_call(population[0], population[0])        
        competition.compete.assert_any_call(population[0], population[1])        
        competition.compete.assert_any_call(population[1], population[0])
        competition.compete.assert_any_call(population[1], population[1])
        
    def test_competition_modification_is_applied_to_population(self):
        network_1 = Network([1, 1])
        network_2 = Network([1, 1])
        
        network_1.synapse_layers[0][0].weight = float(0)
        network_2.synapse_layers[0][0].weight = float(0)
        
        population = [network_1, network_2]
        
        competition = MockCompetition(override_weights=float(1))
        
        epoch = CompetitionEpoch(competition)

        
        epoch.run(PopulationDTO(population))
        
        
        assert population[0].synapse_layers[0][0].weight == 1, f"Expected weight of 1, got {population[0].synapse_layers[0][0].weight}"
        assert population[1].synapse_layers[0][0].weight == 1, f"Expected weight of 1, got {population[1].synapse_layers[0][0].weight}"
        

        
class MockCompetition(Competition):
    def __init__(self, override_weights: Union[float, None] = None):
        self.override_weights = override_weights
    
    def compete(self, challenger: Network, challenged: Network):
        if(self.override_weights != None):
            challenger.synapse_layers[0][0].weight = self.override_weights
            challenged.synapse_layers[0][0].weight = self.override_weights