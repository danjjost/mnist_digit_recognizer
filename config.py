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
        
        self.training_batch_size: int = 100
        self.population_size: int = 100
        self.schema: list[int] = [784, 10]
        
        self.initialization_scale: float = 0.001
        
        self.mnist_testing_folder: str = "./MNIST/testing/"
        self.mnist_training_folder: str = "./MNIST/training/"
        
        self.percent_mutation: float = 0.0005
        self.mutation_step: float = 0.1
        
        self.percent_predation: float = 0.03
        
        self.debug: bool = False
        
        self.is_guessing_percent: float = 0.5
        self.is_guessing_penalty: float = -5