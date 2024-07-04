

import unittest

from src.neuralnet.network import Network
from src.pipeline.population import PopulationDTO
from src.pipeline.population_modifiers.evolver.population_destroyer import PopulationDestroyer
from src.pipeline.population_modifiers.population_generator import PopulationGenerator


class TestPopulationDestroyer(unittest.TestCase):

    def test_deletes_full_population(self):
        population = PopulationGenerator().generate(10, [2, 2, 1])
        destroyer = PopulationDestroyer()
        
        destroyer.destroy(population, number_to_destroy=999999999)
        
        assert len(population.population) == 0
        
        
    def test_deletes_worst_members_of_population(self):
        population = PopulationDTO()
        population.population = []
        
        good_network_1 = Network([1,1,1,1,1])
        good_network_1.score = 100
        
        good_network_2 = Network([1,1,1,1,1])
        good_network_2.score = 100
        
        bad_network_1 = Network([1,1,1,1,1])
        bad_network_1.score = 5
        
        bad_network_2 = Network([1,1,1,1,1])
        bad_network_2.score = 10
        
        population.population.append(good_network_1)
        population.population.append(bad_network_1)
        population.population.append(good_network_2)
        population.population.append(bad_network_2)
        
        destroyer = PopulationDestroyer()
        
        destroyer.destroy(population, number_to_destroy=2)
        
        assert len(population.population) == 2
        assert good_network_1 in population.population
        assert good_network_2 in population.population