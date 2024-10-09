import os

from dotenv import load_dotenv

class AzureConfig:
    def __init__(self):
        load_dotenv()
        
        self.mnist_training_container = "training"
        self.mnist_testing_container = "testing"
        
        self.training_data_blob_connection_string = os.getenv("TRAINING_DATA_BLOB_CONNECTION_STRING")