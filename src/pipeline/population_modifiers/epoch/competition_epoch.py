from src.pipeline.population_modifiers.epoch.i_competition import ICompetition
from src.pipeline.population import PopulationDTO
from src.pipeline.population_modifiers.i_population_modifier import IPopulationModifier


class CompetitionEpoch(IPopulationModifier):
    def __init__(self, competition: ICompetition):
        self.competition = competition

    def run(self, population: PopulationDTO):
        for challenger in population.population:
            for challenged in population.population:
                self.competition.compete(challenger, challenged)
        
        return population