from src.pipeline.population_modifiers.epoch.i_evaluation import IEvaluation
from src.pipeline.population import PopulationDTO
from src.pipeline.population_modifiers.i_population_modifier import IPopulationModifier


class EvaluationEpoch(IPopulationModifier):
    def __init__(self, evaluation: IEvaluation):
        self.evaluation = evaluation

    def run(self, population: PopulationDTO) -> PopulationDTO:
        for network in population.population:
            self.evaluation.evaluate(network)
        
        self.log_population_metrics(population)
        
        return population

    def log_population_metrics(self, population: PopulationDTO):
        try:
            population.average_score = sum([network.score for network in population.population]) / len(population.population)
            min_network = min(population.population, key=lambda network: network.score)
            population.min_score = min_network.score
            max_network = max(population.population, key=lambda network: network.score)
            population.max_score = max_network.score
            print(f'EvaluationEpoch - Evaluation Epoch complete!')
            print(f'EvaluationEpoch - Average score: {population.average_score}')
            print(f'EvaluationEpoch - Min score: {min_network.score}, {min_network.id}')
            print(f'EvaluationEpoch - Max score: {max_network.score}, {max_network.id}')
        except:
            print('EvaluationEpoch - Error calculating population metrics')