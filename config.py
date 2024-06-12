class Config():
    def __init__(self):
        self.decimal_precision: int = 6
        
        self.input_file_path: str = "./populations/population_1.json"
        self.output_file_path: str = "./populations/population_1.json"
        
        self.training_batch_size: int = 32
        self.population_size: int = 25
        self.schema: list[int] = [784, 194, 49, 10]
        
        self.mnist_testing_folder: str = "./MNIST/testing/"
        self.mnist_training_folder: str = "./MNIST/training/"