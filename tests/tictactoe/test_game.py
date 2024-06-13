import unittest
from src.tictactoe.game import Game
from src.tictactoe.random_player import RandomPlayer


class TestGame(unittest.TestCase):
    def test_game(self):
        game = Game()
        game.mute_game()

        game.play(RandomPlayer('X'), RandomPlayer('O'))
        
        self.assertGreaterEqual(len(game.board.history), 5)
        