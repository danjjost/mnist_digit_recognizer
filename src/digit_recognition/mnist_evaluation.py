from decimal import Decimal
from typing import Optional
from config import Config
from src.digit_recognition.batch_loader import BatchLoader, MNISTImage
from src.neuralnet.network import Network
from src.pipeline.evaluation import Evaluation


class MNISTEvaluation(Evaluation):
    def __init__(self, is_test: Optional[bool] = False, config: Optional[Config] = None, batch_loader: Optional[BatchLoader] = None):
        self.config = config or Config()
        self.batch_loader = batch_loader or BatchLoader(self.config, None)
        self.is_test = is_test
        
    def evaluate(self, network: Network):
        print(f'Evaluating network {network.id}.')
        batch = self.get_batch()
        
        network.score = 0
        
        for image in batch:
            self.evaluate_image(network, image)
        
        print(f'Network {network.id} scored {network.score} out of {len(batch)}')
       
        if not self.is_test:
            print(f'Applying gradients to network {network.id}.')
            network.apply_gradients()

    def get_batch(self) -> list[MNISTImage]:
        if self.is_test:
            return self.batch_loader.get_testing_batch(self.config.training_batch_size)
        else:
            return self.batch_loader.get_training_batch(self.config.training_batch_size)
    
    def evaluate_image(self, network: Network, image: MNISTImage):
        network.set_input(image.image_array)
        network.feed_forward()
        outputs = network.get_outputs()
        
        likely_digit = self.get_likely_digit(outputs)
        
        if likely_digit == image.label:
            network.score += 1
        
        if not self.is_test:
            network.back_propagate(self.get_expected_output(image))
        
    def get_likely_digit(self, outputs: list[Decimal]) -> int:
        return outputs.index(max(outputs))
    
    def get_expected_output(self, image: MNISTImage) -> list[Decimal]:
        expected_output = [Decimal(0.0)] * 10
        expected_output[image.label] = Decimal(1.0)
        
        return expected_output