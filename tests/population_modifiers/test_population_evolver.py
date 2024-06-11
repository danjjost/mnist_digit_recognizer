import unittest
from unittest.mock import MagicMock

from src.pipeline.population_generator import PopulationGenerator
from src.pipeline.population_modifiers.population_destroyer import PopulationDestroyer
from src.pipeline.population_modifiers.population_evolver import PopulationEvolver
from src.pipeline.population_modifiers.population_mutator import PopulationMutator
from src.pipeline.population_modifiers.population_rebuilder import PopulationRebuilder


class TestPopulationEvolver(unittest.TestCase):
    def setUp(self) -> None:
        self.population_destroyer = MagicMock(spec=PopulationDestroyer)
        self.population_mutator = MagicMock(spec=PopulationMutator)
        self.population_rebuilder = MagicMock(spec=PopulationRebuilder)

        self.population = PopulationGenerator().generate(10, [2, 2, 1])
        
        self.percent_predation = 0.2
        self.population_evolver = PopulationEvolver(self.percent_predation, self.population_destroyer, self.population_rebuilder, self.population_mutator)
        return super().setUp()
    
    def test_calls_population_destroyer_with_correct_number(self):
        self.population_evolver.run(self.population)

        expected_number_to_replace = 2 # 10 * 20% predation
        self.population_destroyer.destroy.assert_called_once_with(self.population, expected_number_to_replace)
        
    def test_calls_population_rebuilder_with_correct_number(self):
        self.population_evolver.run(self.population)

        expected_number_to_replace = 2 # 10 * 20% predation
        self.population_rebuilder.rebuild.assert_called_once_with(self.population, expected_number_to_replace)
        
    def test_calls_population_mutator(self):
        self.population_evolver.run(self.population)

        self.population_mutator.mutate.assert_called_once_with(self.population)