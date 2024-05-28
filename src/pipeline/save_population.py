from src.pipeline.population import PopulationDTO
from src.pipeline.population_modifier import PopulationModifier


class SavePopulation(PopulationModifier):
    def __init__(self, path: str, population_name: str, population_index: int):
        self.path = path
        self.population_name = population_name
        self.population_index = population_index
        
    def run(self, population: PopulationDTO):
    