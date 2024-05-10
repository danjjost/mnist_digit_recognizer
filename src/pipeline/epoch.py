from typing import List
from src.neuralnet.network import Network
from src.pipeline.competition import Competition


class Epoch():
    def __init__(self, competition: Competition):
        self.competition = competition

    def run(self, population: list[Network]):
        for challenger in population:
            for challenged in population:
                self.competition.compete(challenger, challenged)
        
        return population