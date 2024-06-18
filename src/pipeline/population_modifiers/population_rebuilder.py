from random import Random
import uuid
from src.neuralnet.network import Network
from src.neuralnet.network_to_dict import NetworkToDict
from src.pipeline.population import PopulationDTO


class PopulationRebuilder():
    
    def __init__(self, random: Random = Random()):
        self.random = random
        
    def rebuild(self, population: PopulationDTO, number_to_copy: int) -> None:
        print(f'PopulationRebuilder - Rebuilding {number_to_copy} individuals from population of {len(population.population)} individuals.')
        for _ in range(number_to_copy):
            network = self.random.choice(population.population)
            cloned_network = self.deep_clone(network)
            population.population.append(cloned_network)
            
    def deep_clone(self, network: Network):
        dictionary = NetworkToDict().to_dict(network) 
        new_network = NetworkToDict().from_dict(dictionary)
        new_network.id = str(uuid.uuid4())
        return new_network