import time
import unittest
from unittest.mock import MagicMock

from config import Config
from src.azure_helpers.blob_client import BlobClient
from src.azure_helpers.blob_poller import BlobPoller
from src.neuralnet.network import Network
from src.neuralnet.to_dict.network_to_dict import NetworkToDict
from src.pipeline.population_modifiers.population_generator import PopulationGenerator


class TestBlobPoller(unittest.TestCase):
    def setUp(self) -> None:
        self.blob_client = MagicMock(spec=BlobClient)
        self.blob_poller = BlobPoller(self.blob_client)

    def test_updates_network_with_updates_from_blob_storage_output(self):
        population = PopulationGenerator().generate(2, [1,1])
        
        modified_network_1 = NetworkToDict().clone(population.population[0])
        modified_network_1.node_layers[0][0].bias = 2.3
        
        modified_network_2 = NetworkToDict().clone(population.population[1])
        modified_network_2.node_layers[0][0].bias = 2.4
        
        self.blob_client.get_all_ids.return_value = [modified_network_1.id, modified_network_2.id]
        self.mock_blob_download(modified_network_1.id, modified_network_1)
        self.mock_blob_download(modified_network_2.id, modified_network_2)

        
        self.blob_poller.poll(population)
        
        
        self.assertEqual(population.population[0].node_layers[0][0].bias, 2.3)
        self.assertEqual(population.population[1].node_layers[0][0].bias, 2.4)

        
    def test_update_polling_is_cancelled_after_timeout_expires(self):
        config = Config()
        config.poll_timeout_ms = 250
        config.polling_rate_ms = 5
        population = PopulationGenerator().generate(2, [1,1])
        self.blob_poller = BlobPoller(self.blob_client, config)
        
        
        time_started = self.get_current_time_ms()

        self.blob_poller.poll(population)

        time_finished = self.get_current_time_ms()
        

        timeElapsed = time_finished - time_started
        self.assertGreaterEqual(timeElapsed, 250)
        self.assertLessEqual(timeElapsed, 500)
    
    def get_current_time_ms(self):
        return time.time() * 1000
    
    def mock_blob_download(self, id: str, network: Network):
        if hasattr(self, 'blob_download_dictionary') is False:
            self.blob_download_dictionary: dict[str, Network] = dict()

        self.blob_download_dictionary[id] = network        
        
        self.blob_client.download.side_effect = self.generate_side_effect()
        
    def generate_side_effect(self):
        def side_effect(id: str):
            return self.blob_download_dictionary[id]
        
        return side_effect
        