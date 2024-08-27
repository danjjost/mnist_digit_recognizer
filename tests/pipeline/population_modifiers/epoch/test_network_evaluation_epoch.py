import unittest
from unittest.mock import MagicMock

from src.azure.blob_client import BlobClient
from src.azure.blob_poller import BlobPoller
from src.pipeline.population import PopulationDTO
from src.pipeline.population_modifiers.epoch.network_evaluation.network_evaluation_epoch import NetworkEvaluationEpoch
from src.pipeline.population_modifiers.population_generator import PopulationGenerator


class TestNetworkEvaluationEpoch(unittest.TestCase):
    def setUp(self) -> None:
        self.input_blob_client = MagicMock(spec=BlobClient)
        self.output_blob_client = MagicMock(spec=BlobClient)
        
        self.blob_poller = MagicMock(spec=BlobPoller)
        
        self.network_evaluation_epoch = NetworkEvaluationEpoch(self.input_blob_client, self.output_blob_client, self.blob_poller)
        
    def test_uploads_blobs_in_population(self):
        population = PopulationGenerator().generate(5, [2, 2, 1])
        
        
        self.network_evaluation_epoch.run(population)
        
        
        self.input_blob_client.upload_batch.assert_called_once_with(population.population)
    
    def test_polls_blobs(self):
        population = PopulationGenerator().generate(5, [2, 2, 1])
        
        self.network_evaluation_epoch.run(population)
        
        self.blob_poller.poll.assert_called_once_with(population)
        
    def test_returns_population_modified_by_polling(self):
        population = PopulationGenerator().generate(5, [2, 2, 1])
        
        self.blob_poller.poll.side_effect = self.modify_population
        
        result = self.network_evaluation_epoch.run(population)
        
        
        for individual in result.population:
            self.assertEqual(individual.id, "modified id")
            
    def test_calls_clear_on_input_and_output_blob_clients(self):
        population = PopulationGenerator().generate(5, [2, 2, 1])
        
        self.network_evaluation_epoch.run(population)
        
        self.input_blob_client.clear.assert_called_once()
        self.output_blob_client.clear.assert_called_once()
    
    def test_returns_modified_population(self):
        population = PopulationGenerator().generate(5, [2, 2, 1])
        
        self.blob_poller.poll.side_effect = self.modify_population
        
        result = self.network_evaluation_epoch.run(population)
        
        self.assertEqual(population, result)
        
        for individual in result.population:
            self.assertEqual(individual.id, "modified id")
    
    def modify_population(self, population: PopulationDTO) -> None:
        for individual in population.population:
            setattr(individual, "id", "modified id")