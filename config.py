from enum import Enum, auto


class NetworkEvaluationMode(Enum):
    TRAIN = auto()
    TEST = auto()

class Config():
    def __init__(self):
        self.mode: NetworkEvaluationMode = NetworkEvaluationMode.TRAIN
        
        self.input_file_path: str = "./populations/population_1.json"
        self.output_file_path: str = "./populations/population_1.json"
        self.learning_rate: float = 0.0005
        
        self.training_batch_size: int = 32
        self.population_size: int = 1
        self.schema: list[int] = [784, 10]
        
        self.initialization_scale: float = 0.01
        
        self.mnist_testing_folder: str = "./MNIST/testing/"
        self.mnist_training_folder: str = "./MNIST/training/"
        
        self.debug: bool = False