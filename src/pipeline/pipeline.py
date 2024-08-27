from src.pipeline.population import PopulationDTO
from src.pipeline.population_modifiers.i_population_modifier import IPopulationModifier


class Pipeline():
    def __init__(self):
        self.pipeline: list[IPopulationModifier] = []
    
    def add(self, population_modifier: IPopulationModifier):
        self.pipeline.append(population_modifier)
    
    def run(self, population: PopulationDTO) -> PopulationDTO:
        previous_population = population
        
        for pipeline_modifier in self.pipeline:
            previous_population = pipeline_modifier.run(previous_population)
            
        return previous_population