from src.tictactoe.board import Board
from src.tictactoe.player import Player


class RandomPlayer(Player):
    def __init__(self, symbol: str):
        super().__init__(symbol)
        
    def get_move(self, board):
        import random
        return random.choice(self.get_available_moves(board))
    
    def get_available_moves(self, board: Board):
        return [i for i, x in enumerate(board.current) if x == '']