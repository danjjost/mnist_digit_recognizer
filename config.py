from enum import Enum, auto


class NetworkEvaluationMode(Enum):
    TRAIN = auto()
    TEST = auto()

class Config():
    def __init__(self):
        self.mode = NetworkEvaluationMode.TRAIN
        
        self.input_file_path = "./populations/population20_1356_10.json"
        self.output_file_path = "./populations/population20_1356_10.json"

        self.mnist_testing_folder = "./MNIST/testing/"
        self.mnist_training_folder = "./MNIST/training/"

        # The learning rate of the network, affects the speed of gradient descent.
        # A higher learning rate may be best at the beginning of training, 
        # and a lower learning rate may be best at the end.
        # Default: 0.001 
        self.learning_rate: float = 0.001
        
        # The number of training batches to run before evaluating the network and applying gradients
        self.training_batch_size: int = 32
        
        # The number of networks in the population.
        self.population_size: int = 20
        
        # The number of nodes in each layer of the network
        self.schema: list[int] = [1352, 10]
        
        # The starting scale of the weights and biases of the network
        self.initialization_scale: float = 0.01
        
        # Mutation percent is the chance that a node or synapse will be mutated
        # Mutation step is the scale that a node or synapse will be mutated by. 
        # For example, 0.1 would add or subtract a maximum of 0.1 from the node or synapse
        self.percent_mutation: float = 0.01
        self.mutation_step: float = 0.003
        
        # The percent of the population that will be deleted and replaced each generation by the population evolver
        self.percent_predation: float = 0.05
        
        # If True, logging will output verbosely
        self.debug: bool = False
        
        # What percentage of the population to consider the 'top percentile'
        # This is used for selecting the best networks to clone after the worst performing networks are deleted.
        self.top_percentile: float = 0.1