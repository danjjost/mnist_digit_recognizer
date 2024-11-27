from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from config import Config
from src.digit_recognition.mnist_image_evaluator import MNISTImageEvaluator
from src.pipeline.population_modifiers.epoch.i_evaluation import IEvaluation
from src.pipeline.population import PopulationDTO
from src.pipeline.population_modifiers.i_population_modifier import IPopulationModifier

class ParallelEvaluationEpoch(IPopulationModifier):
    def __init__(self, evaluation: IEvaluation, config: Optional[Config] = None):
        self.evaluation = evaluation
        self.config = config or Config()

    def run(self, population: PopulationDTO) -> PopulationDTO:
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.evaluation.evaluate, network) for network in population.population]
            
            for future in futures:
                future.result()  
        
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

            if self.config.debug:
                print(f'EvaluationEpoch - Min score: {min_network.score}, {min_network.id}')
                print(f'EvaluationEpoch - Max score: {max_network.score}, {max_network.id}')

                for i in range(10):
                    print(f'EvaluationEpoch - Score for digit {i + 1}: {MNISTImageEvaluator.scores[i]}')
            print(f'EvaluationEpoch - Scores: {MNISTImageEvaluator.scores}')
        except:
            print('EvaluationEpoch - Error calculating population metrics')
