import threading
from typing import Optional
from config import Config, NetworkEvaluationMode
from src.digit_recognition.MNISTImage import MNISTImage
from src.neuralnet.network import Network


class MNISTImageEvaluator:
    scores = [0] * 10  
    scores_lock = threading.Lock()
    
    def __init__(self, config: Optional[Config]) -> None:
        self.config: Config = config or Config()
        self.mode: NetworkEvaluationMode = self.config.mode
    
    def evaluate_image(self, network: Network, image: MNISTImage):        
        network.set_input(image.image_array + image.convolved_image_array)
        network.feed_forward()
        outputs = network.get_outputs()
        
        likely_digit = self.get_likely_digit(outputs)
        if self.config.debug: 
            print(f'MNISTImageEvaluator - Network predicted the number "{likely_digit}", was actually "{image.label}".')
        
        if likely_digit == image.label:
            network.score += 1
        #    with MNISTImageEvaluator.scores_lock:
        #        MNISTImageEvaluator.scores[likely_digit] += 1
        
        if self.mode == NetworkEvaluationMode.TRAIN:
            if self.config.debug: 
                print(f'MNISTImageEvaluator - Back propagating.')
            network.back_propagate(self.get_expected_output(image))
        
        return likely_digit
        
    def get_likely_digit(self, outputs: list[float]) -> int:
        return outputs.index(max(outputs))
    
    def get_expected_output(self, image: MNISTImage) -> list[float]:
        expected_output = [0.0] * 10
        expected_output[image.label] = 1.0
        
        return expected_output