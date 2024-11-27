from typing import Optional
from config import Config, NetworkEvaluationMode
from src.digit_recognition.image_loader import ImageLoader, MNISTImage
from src.digit_recognition.mnist_image_evaluator import MNISTImageEvaluator
from src.neuralnet.network import Network
from src.pipeline.population_modifiers.epoch.i_evaluation import IEvaluation


class MNISTEvaluation(IEvaluation):
    
    def __init__(self, 
        config: Optional[Config] = None, 
        batch_loader: Optional[ImageLoader] = None,
        mnist_image_evaluator: Optional[MNISTImageEvaluator] = None
    ):
        self.config = config or Config()
        self.batch_loader = batch_loader or ImageLoader(self.config, None)
        self.mnist_image_evaluator = mnist_image_evaluator or MNISTImageEvaluator(self.config)
        
        
    def evaluate(self, network: Network):
        network.score = 0
        
        likely_digits: list[int] = []
        
        for _ in range(self.config.training_batch_size):
            image = self.get_image()
            likely_digit = self.mnist_image_evaluator.evaluate_image(network, image)
            likely_digits.append(likely_digit)

        self.apply_gradients_if_training(network)
            
        if self.config.debug:
            print(f'MNISTEvaluation - Network scored {network.score}/{self.config.training_batch_size}.')

    def apply_gradients_if_training(self, network: Network):
        if self.config.mode == NetworkEvaluationMode.TRAIN:
            if self.config.debug:
                print("MNISTEvaluation - Applying Gradients")
            
            network.apply_gradients()
    
    def get_image(self) -> MNISTImage:
        if self.config.mode == NetworkEvaluationMode.TEST:
            return self.batch_loader.get_testing_image()
        else:
            return self.batch_loader.get_training_image()