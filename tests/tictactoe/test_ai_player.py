import unittest
from unittest.mock import MagicMock
from src.neuralnet.network import Network
from src.tictactoe.ai_player import AiPlayer
from src.tictactoe.board import Board


class TestAiPlayer(unittest.TestCase):
    def test_get_move_raises_value_exception_if_network_not_set(self):
        player = AiPlayer('X')
        board = Board()
        
        with self.assertRaises(ValueError):
            player.get_move(board)


    def test_set_network_sets_network(self):
        player = AiPlayer('X')
        
        network = Network([1])
        
        player.set_network(network)

        assert player.network == network
        
        
    def test_get_move_returns_highest_probable_move_for_x(self):
        player = AiPlayer('X')
        
        network = MagicMock(spec=Network)
        network.get_outputs.return_value = [0, 0, 0.9, 0, 0, 0, 0, 0, 0]
        
        player.set_network(network)

        move = player.get_move(Board())
        
        assert move == 2
        
        
    def test_get_move_returns_lowest_probable_move_for_0(self):
        player = AiPlayer('O')
        
        network = MagicMock(spec=Network)
        network.get_outputs.return_value = [1, 0.2, 1, 1, 1, 1, 1, 1, 1]
        
        player.set_network(network)

        move = player.get_move(Board())
        
        assert move == 1