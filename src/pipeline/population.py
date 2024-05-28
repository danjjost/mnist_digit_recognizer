from src.neuralnet.network import Network


class PopulationDTO():
    def __init__(self, population: list[Network] | None = None) -> None:
        self.population = population if population != None else []