import time
import unittest
from azure_config import AzureConfig
from config import Config
from src.azure_helpers.blob_client import BlobClient
from src.azure_helpers.blob_poller import BlobPoller
from src.neuralnet.to_dict.network_to_dict import NetworkToDict
from src.pipeline.population_modifiers.population_generator import PopulationGenerator
from src.utilities.container_client_builder import ContainerClientBuilder

class TestBlobPoller(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config()
        self.azure_config = AzureConfig()
        
        self.network_to_dict = NetworkToDict(self.config)
        
        self.__initialize_container_client()
        
        self.blob_client = BlobClient(self.network_to_dict, self.test_container_client)
        self.blob_poller = BlobPoller(self.blob_client)
        
    def __initialize_container_client(self):
        self.test_container_client = ContainerClientBuilder().build(self.azure_config, "test-blob-poller")
        
        if self.test_container_client.exists():
            self.test_container_client.delete_container()
            time.sleep(10)

        self.__create_container()
        self.__query_container()

    def __create_container(self):
        container_creation_tries: int = 0
        while container_creation_tries < 10:
            try:
                self.test_container_client.create_container()
                return
            except:
                time.sleep(10)
            
            container_creation_tries += 1
        raise Exception("Could not create container, failed after too many tries")
    
    def __query_container(self):
        container_query_tries: int = 0
        
        while container_query_tries < 10:
            try:
                self.test_container_client.get_container_properties()
                return
            except:
                time.sleep(10)
            
            container_query_tries += 1
            
        raise Exception("Could not query container, failed after too many tries")
                        
    def test_can_update_population_from_blob_storage(self):
        population = PopulationGenerator().generate(2, [1,1])
        
        modified_network_1 = NetworkToDict().clone(population.population[0])
        modified_network_1.node_layers[0][0].bias = 2.3
        
        modified_network_2 = NetworkToDict().clone(population.population[1])
        modified_network_2.node_layers[0][0].bias = 2.4
                
        self.blob_client.upload_blob(modified_network_1)
        self.blob_client.upload_blob(modified_network_2)
        
        
        self.blob_poller.poll(population)
        
        
        self.assertEqual(population.population[0].node_layers[0][0].bias, 2.3)
        self.assertEqual(population.population[1].node_layers[0][0].bias, 2.4)
        