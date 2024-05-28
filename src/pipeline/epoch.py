from src.pipeline.competition import Competition
from src.pipeline.population import PopulationDTO
from src.pipeline.population_modifier import PopulationModifier


class Epoch(PopulationModifier):
    def __init__(self, competition: Competition):
        self.competition = competition

    def run(self, population: PopulationDTO):
        for challenger in population.population:
            for challenged in population.population:
                self.competition.compete(challenger, challenged)
        
        return population