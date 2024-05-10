from src.neuralnet.network import Network


class Population():
    def __init__(self) -> None:
        self.population: list[Network] = []