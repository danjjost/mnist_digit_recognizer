
from typing import Optional

from src.pipeline.population import PopulationDTO
from src.pipeline.population_modifier import PopulationModifier
from src.pipeline.population_modifiers.population_destroyer import PopulationDestroyer
from src.pipeline.population_modifiers.population_mutator import PopulationMutator
from src.pipeline.population_modifiers.population_rebuilder import PopulationRebuilder


class PopulationEvolver(PopulationModifier):
    
    def __init__(self, 
            percent_predation: float = 0.1,
            population_destroyer: Optional[PopulationDestroyer] = None, 
            population_rebuilder: Optional[PopulationRebuilder] = None,
            population_mutator: Optional[PopulationMutator] = None):
        self.percent_predation = percent_predation or 0.1
        
        self.population_destroyer = population_destroyer or PopulationDestroyer()
        self.population_rebuilder = population_rebuilder or PopulationRebuilder()
        self.population_mutator = population_mutator or PopulationMutator()
    
    def run(self, population: PopulationDTO) -> PopulationDTO:
        number_to_replace = int(len(population.population) * self.percent_predation)

        if number_to_replace == 0:
            number_to_replace = 1

        self.population_destroyer.destroy(population, number_to_replace)
        self.population_rebuilder.rebuild(population, number_to_replace)
        self.population_mutator.mutate(population)
        
        return population