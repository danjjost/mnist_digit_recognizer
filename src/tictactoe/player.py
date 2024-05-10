import uuid

from src.tictactoe.board import Board


class Player:
    def __init__(self, symbol: str):
        if symbol == "":
            raise ValueError('Symbol cannot be empty')
        
        self.symbol = symbol
        self.guid = uuid.uuid4()

    def get_move(self, board: Board) -> int:
        return 0