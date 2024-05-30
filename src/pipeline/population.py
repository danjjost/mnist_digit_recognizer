import uuid
from src.neuralnet.network import Network


class PopulationDTO():
    def __init__(self, population: list[Network] = []) -> None:
        self.id = str(uuid.uuid4())
        self.population: list[Network] = population
        
    def clear(self) -> None:
        self.population = []