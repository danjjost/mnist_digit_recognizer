
import unittest
from unittest.mock import MagicMock

from src.file.file_loader import FileLoader
from src.neuralnet.network import Network
from src.neuralnet.network_to_dict import NetworkToDict
from src.pipeline.load_population import LoadPopulation
from src.pipeline.population import PopulationDTO
from src.pipeline.save_population import SavePopulation


class TestLoadPopulation(unittest.TestCase):
    
    def setUp(self) -> None:
        self.path = "path"
        self.file_loader = MagicMock(spec=FileLoader)
        self.network_to_dict = NetworkToDict()
        self.load_population = LoadPopulation(self.path, file_loader=self.file_loader, network_to_dict=self.network_to_dict)
    
    def test_load_population_calls_file_loader_with_path(self):
        self.file_loader.load.return_value = ""
        
        self.load_population.run(PopulationDTO())

        self.file_loader.load.assert_called_once_with(self.path)
        
    def test_load_population_clears_population(self):
        population = PopulationDTO()
        population.population = [MagicMock(), MagicMock()]
        
        self.file_loader.load.return_value = ""
        
        self.load_population.run(population)
        
        self.assertEqual(population.population, [])
        
    def test_load_population_returns_population_parsed_from_population_file_json(self):
        network1 = Network([1,2,3])
        network1.node_layers[2][2].bias = float(1.3)
        
        network2 = Network([2,2])
        network2.synapse_layers[0][1].weight = float(1.2)
        
        population = PopulationDTO([network1, network2])
        population_json = SavePopulation('').get_json(population)
        
        
        self.file_loader.load.return_value = population_json
        
        
        loaded_population = self.load_population.run(PopulationDTO())
        
        assert len(loaded_population.population) == 2
        assert len(loaded_population.population[0].node_layers) == 3
        assert len(loaded_population.population[1].synapse_layers) == 1
        assert loaded_population.population[0].node_layers[2][2].bias == float(1.3)
        assert loaded_population.population[1].synapse_layers[0][1].weight == float(1.2)
        
        
        