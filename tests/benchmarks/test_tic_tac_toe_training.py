import unittest
from src.neuralnet.network import Network
from src.pipeline.tic_tac_toe_training.board_evaluator import BoardEvaluator
from src.pipeline.tic_tac_toe_training.board_trainer import BoardTrainer
from src.tictactoe.ai_player import AiPlayer
from src.tictactoe.board import Board


class TestTicTacToeTraining(unittest.TestCase):
    
    # Configuration
    def setUp(self):
        self.network_dimensions = [9, 27, 81, 9]
    
    
    # Final Board:
    #   |   | 
    # -------
    # X | X | X
    # -------
    # O | O |
    
    
    # This benchmark simulates a single game of tic-tac-toe.
    # The network has a chance to learn from the game, and we can then assert that it will pick the final move correctly.
    
    # This benchmark is inherently overfitting, as it is training on a single game. 
    # The test's purpose is to ensure that the training process works at the simplest level.
    
 #   def test_training_for_single_problem(self):
        board = Board()
        
        board.move(3, 'X')
        board.move(6, 'O')
        board.move(4, 'X')
        board.move(7, 'O')
        board.move(5, 'X')
                
        board_evaluator = BoardEvaluator()


        networkX = Network(self.network_dimensions)
        networkO = Network(self.network_dimensions)
        
        for _ in range(1000):
            BoardTrainer(board_evaluator).train(board, networkX)
            BoardTrainer(board_evaluator).train(board, networkO)


        playerX = AiPlayer('X')
        playerX.set_network(networkX)
        
        testBoard = Board()
        testBoard.move(3, 'X')
        testBoard.move(6, 'O')
        testBoard.move(4, 'X')
        testBoard.move(7, 'O')
        
        move = playerX.get_move(testBoard)
        
        assert move == 5