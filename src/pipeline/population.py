from src.neuralnet.network import Network


class PopulationDTO():
    def __init__(self, population: list[Network] = []) -> None:
        self.population = population