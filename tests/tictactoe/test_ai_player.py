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
        
        
    def test_get_move_for_O_does_not_return_invalid_move(self):
        player = AiPlayer('O')
        
        network = MagicMock(spec=Network)
        network.get_outputs.return_value = [0.5, 0.2, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        
        player.set_network(network)

        board = Board()
        board.move(2, 'X')

        move = player.get_move(board)
        
        assert move != 2 # 2 is already taken
        assert move == 1 # 1 is the next best move
    
    
    def test_get_move_for_X_does_not_return_invalid_move(self):
        player = AiPlayer('X')
        
        network = MagicMock(spec=Network)
        network.get_outputs.return_value = [0.5, 0.8, 0.7, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        
        player.set_network(network)

        board = Board()
        board.move(1, 'X')

        move = player.get_move(board)
        
        assert move != 1 # 1 is already taken
        assert move == 2 # 2 is the next best move
        
        
    def test_throws_if_no_moves_available(self):
        player = AiPlayer('X')

        player.set_network(Network([1]))
        
        board = Board()
        board.move(0, 'X')
        board.move(1, 'O')
        board.move(2, 'X')
        board.move(3, 'O')
        board.move(4, 'X')
        board.move(5, 'O')
        board.move(6, 'X')
        board.move(7, 'O')
        board.move(8, 'X')
        
        with self.assertRaises(ValueError):
            player.get_move(board)