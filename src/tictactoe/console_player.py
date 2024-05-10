

from src.tictactoe.player import Player


class ConsolePlayer(Player):
    def __init__(self, sumbol):
        super().__init__(sumbol)

    def get_move(self, board):
        while True:
            try:
                move = int(input(f"Enter move for {self.symbol}: "))
                if move not in self.get_available_moves(board):
                    raise ValueError
                return move
            except ValueError:
                print("Invalid move. Try again.")
                
    def get_available_moves(self, board):
        return [i for i, x in enumerate(board) if x == ' ']