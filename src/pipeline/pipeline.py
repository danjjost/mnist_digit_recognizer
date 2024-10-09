from config import Config
from src.digit_recognition.mnist_image_evaluator import MNISTImageEvaluator
from src.pipeline.population import PopulationDTO
from src.pipeline.population_modifiers.i_population_modifier import IPopulationModifier


class Pipeline():
    def __init__(self):
        self.total_score = 0
        self.total_possible_score = 0
        self.pipeline: list[IPopulationModifier] = []
    
    def add(self, population_modifier: IPopulationModifier):
        self.pipeline.append(population_modifier)
    
    def run(self, population: PopulationDTO) -> PopulationDTO:
        previous_population = population
        
        for pipeline_modifier in self.pipeline:
            previous_population = pipeline_modifier.run(previous_population)
        
        self.total_score += previous_population.average_score
        self.total_possible_score += Config().training_batch_size
        
        print(f"Running score: {self.total_score}/{self.total_possible_score}")
        print(f"Running percent correct: {(0.0 + self.total_score)/self.total_possible_score}")
        
        #with MNISTImageEvaluator.scores_lock:
        #    print("Correct predictions per digit:")
        #    for digit in range(10):
        #        score = MNISTImageEvaluator.scores[digit]
        #        print(f"Digit {digit}: {score} correct predictions")
        
        return previous_population