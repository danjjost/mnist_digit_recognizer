import json
import unittest
from unittest.mock import MagicMock

from src.file.file_writer import FileWriter
from src.neuralnet.network_to_dict import NetworkToDict
from src.pipeline.population import PopulationDTO
from src.pipeline.save_population import SavePopulation

class TestSavePopulation(unittest.TestCase):

    def setUp(self) -> None:
        self.path = "./test_population_1.json"
        
        self.mock_network_to_dict = MagicMock(spec=NetworkToDict)
        self.mock_network_to_dict.to_dict.return_value = {"id": "1"} 
        
        self.mock_file_writer = MagicMock(spec=FileWriter)
        
        self.save_population = SavePopulation(self.path, self.mock_network_to_dict, self.mock_file_writer)


    def test_run_calls_network_to_dict_for_each_network(self):        
        population = PopulationDTO()
        population.population = [MagicMock(), MagicMock()] # type: ignore
        
        self.save_population.run(population)
        
        self.mock_network_to_dict.to_dict.assert_any_call(population.population[0]) # type: ignore
        self.mock_network_to_dict.to_dict.assert_any_call(population.population[1]) # type: ignore
        
    def test_run_calls_file_writer_with_correct_parameters(self):
        population = PopulationDTO()
        population.population = [MagicMock(), MagicMock()] # type: ignore
        
        self.mock_network_to_dict.to_dict.side_effect = [{"id": "1"}, {"id": "2"}]
        
        
        self.save_population.run(population)
        
        
        expected_json = json.dumps([{"id": "1"}, {"id": "2"}])
        
        
        self.mock_file_writer.save.assert_any_call(self.path, expected_json)