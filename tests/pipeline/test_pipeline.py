import unittest
from unittest.mock import Mock

from src.neuralnet.network import Network
from src.pipeline.pipeline import Pipeline
from src.pipeline.population import Population
from src.pipeline.population_modifier import PopulationModifier

class TestPipeline(unittest.TestCase):
    def test_pipeline_calls_population_modifier_with_population(self):
        pipeline = Pipeline()
        
        population_modifier = Mock()
        pipeline.add(population_modifier)
        
        population = Population()

        
        pipeline.run(population)

        
        population_modifier.run.assert_called_once_with(population)

    def test_pipeline_returns_the_output_population_from_modifier(self):
        pipeline = Pipeline()
        
        population_modifier = Mock()
        pipeline.add(population_modifier)
        
        input_population = Population()
        output_population = Population()
        
        population_modifier.run.return_value = output_population
        
        
        result = pipeline.run(input_population)
        
        
        assert result == output_population, f"Expected {output_population}, got {result}"
        
    def test_pipeline_chains_multiple_population_modifiers(self):
        pipeline = Pipeline()
        
        population_modifier_1 = Mock(spec=PopulationModifier)
        population_modifier_2 = Mock(spec=PopulationModifier)
        pipeline.add(population_modifier_1)
        pipeline.add(population_modifier_2)
        
        input_population = Population()
        intermediate_population = Population()
        output_population = Population()
        
        population_modifier_1.run.return_value = intermediate_population
        population_modifier_2.run.return_value = output_population
        
        
        result = pipeline.run(input_population)
        
        
        assert result == output_population, f"Expected {output_population}, got {result}"
        population_modifier_1.run.assert_called_once_with(input_population)