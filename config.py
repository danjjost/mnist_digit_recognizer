class Config():
    def __init__(self):
        self.input_file_path: str = "./populations/population_1.json"
        self.output_file_path: str = "./populations/population_1.json"
        
        self.training_batch_size: int = 15
        self.population_size: int = 25
        self.schema: list[int] = [784, 194, 49, 10]
        
        self.mnist_testing_folder: str = "./MNIST/testing/"
        self.mnist_training_folder: str = "./MNIST/training/"