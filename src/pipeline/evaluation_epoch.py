from src.pipeline.evaluation import Evaluation
from src.pipeline.population import PopulationDTO
from src.pipeline.population_modifier import PopulationModifier


class EvaluationEpoch(PopulationModifier):
    def __init__(self, evaluation: Evaluation):
        self.evaluation = evaluation

    def run(self, population: PopulationDTO) -> PopulationDTO:
        for network in population.population:
            self.evaluation.evaluate(network)
        
        self.log_population_metrics(population)
        
        return population

    def log_population_metrics(self, population: PopulationDTO):
        try:
            population.average_score = sum([network.score for network in population.population]) / len(population.population)
            population.min_score = min([network.score for network in population.population])
            population.max_score = max([network.score for network in population.population])
            print(f'Evaluation Epoch complete!')
            print(f'Average score: {average_score}')
            print(f'Min score: {min_score}')
            print(f'Max score: {max_score}')
        except:
            print('Error calculating population metrics')