from src.pipeline.population import Population
from src.pipeline.population_modifier import PopulationModifier


class Pipeline():
    def __init__(self):
        self.pipeline = []
    
    def add(self, population_modifier: PopulationModifier):
        self.pipeline.append(population_modifier)
    
    def run(self, population: Population) -> Population:
        previous_population = population
        
        for pipeline_modifier in self.pipeline:
            previous_population = pipeline_modifier.run(previous_population)
            
        return previous_population