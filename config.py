from enum import Enum, auto


class NetworkEvaluationMode(Enum):
    TRAIN = auto()
    TEST = auto()

class Config():
    def __init__(self):
        self.mode: NetworkEvaluationMode = NetworkEvaluationMode.TRAIN
        
        self.input_file_path: str = "./populations/population10_2136_10.json"
        self.output_file_path: str = "./populations/population10_2136_10.json"
        
        self.mnist_testing_folder: str = "./MNIST/testing/"
        self.mnist_training_folder: str = "./MNIST/training/"

        # The learning rate of the network, affects the speed of gradient descent
        self.learning_rate: float = 0.01 # Default: 0.001 
        
        # The number of training batches to run before evaluating the network and applying gradients
        self.training_batch_size: int = 32
        
        # The number of networks in the population, this may become problematic if the population is less than 10
        self.population_size: int = 10
        
        # The number of nodes in each layer of the network
        self.schema: list[int] = [2136, 10]
        
        # The starting scale of the weights and biases of the network
        self.initialization_scale: float = 0.01
        
        # Mutation percent is the chance that a node or synapse will be mutated
        # Mutation step is the scale that a node or synapse will be mutated by. For example, 0.1 would add or subtract a maximum of 0.1 from the node or synapse
        self.percent_mutation: float = 0.01
        self.mutation_step: float = 0.003
        
        # The percent of neurons/synapses that will be crossed over
        self.cross_percent: float = 0.02
        
        # The percent of the population that will be killed off and replaced each generation by the population evolver
        self.percent_predation: float = 0.01
        
        # If True, logging will output verbosely
        self.debug: bool = False
        
        # If a network guesses a single value more than the is_guessing_percent of the time, 
        # it will be penalized by is_guessing_penalty
        self.is_guessing_percent: float = 0.90
        self.is_guessing_penalty: float = 0
        
        # What percentage of the population to consider the 'top percentile'
        # This is used for selecting the best networks to cross
        self.top_percentile: float = 0.1
        
        # For use with the remote evaluation epoch
        # After this amount of time has passed, any non-updated blobs will remain non-updated and the process will continue.
        self.poll_timeout_ms: float = 60 * 60 * 1000 # 60 minutes
        
        # For use with the remote evaluation epoch
        # The amount of time to sleep between queries to blob storage for updated blobs
        self.polling_rate_ms: float = 10 * 1000 # 10 seconds