import unittest

from src.pipeline.tic_tac_toe_training.board_evaluator import BoardEvaluator
from src.pipeline.tic_tac_toe_training.board_training_case import BoardTrainingCase
from src.tictactoe.board import Board


class TestBoardEvaluator(unittest.TestCase):
    def test_board_states_are_returned(self):
        board = self.get_simple_board()

        board_evaluator = BoardEvaluator()
        
        training_cases = board_evaluator.get_training_cases(board)

        self.verify_simple_board_training_cases(training_cases)
        
        
    def test_board_states_for_O_victory(self):
        board = self.get_O_victory_board()

        board_evaluator = BoardEvaluator()
        
        training_cases = board_evaluator.get_training_cases(board)
        
        self.verify_O_victory_board_training_cases(training_cases)


    def test_board_states_for_no_winner(self):
        board = self.get_no_winner_board()

        board_evaluator = BoardEvaluator()
        
        training_cases = board_evaluator.get_training_cases(board)
        
        assert len(training_cases) == 0
        
        
    def get_no_winner_board(self):
        # O | O | X
        # X | X | O
        # O | X | X

        board = Board()
        board.move(2, "X")
        board.move(0, "O")
        
        board.move(3, "X")
        board.move(1, "O")
        
        board.move(4, "X")
        board.move(5, "O")
        
        board.move(7, "X")
        board.move(6, "O")
        
        board.move(8, "X")
        
        return board

    def verify_O_victory_board_training_cases(self, training_cases: list[BoardTrainingCase]):
        assert len(training_cases) == 6
        
        assert training_cases[0].input_board == [0.5, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        assert training_cases[0].move_index == 1
        assert training_cases[0].target_value == 0.5
        
        assert training_cases[1].input_board == [0, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        assert training_cases[1].move_index == 0
        assert training_cases[1].target_value == 0
        
        assert training_cases[2].input_board == [0, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        assert training_cases[2].move_index == 2
        assert training_cases[2].target_value == 0.5
        
        assert training_cases[3].input_board == [0, 1, 1, 0.5, 0, 0.5, 0.5, 0.5, 0.5]
        assert training_cases[3].move_index == 4
        assert training_cases[3].target_value == 0
        
        assert training_cases[4].input_board == [0, 1, 1, 1, 0, 0.5, 0.5, 0.5, 0.5]
        assert training_cases[4].move_index == 3
        assert training_cases[4].target_value == 0.5
        
        assert training_cases[5].input_board == [0, 1, 1, 1, 0, 0.5, 0.5, 0.5, 0]
        assert training_cases[5].move_index == 8
        assert training_cases[5].target_value == 0
        

    def get_O_victory_board(self):
        # O | X | X
        # X | O |
        #   |   | O

        board = Board()
        board.move(1, "X")
        board.move(0, "O")
        
        board.move(2, "X")
        board.move(4, "O")
        
        board.move(3, "X")
        board.move(8, "O")
        
        return board
    

    def verify_simple_board_training_cases(self, training_cases: list[BoardTrainingCase]):
        assert len(training_cases) == 5
        
        assert training_cases[0].input_board == [1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        assert training_cases[0].move_index == 0
        assert training_cases[0].target_value == 1
        
        assert training_cases[1].input_board == [1, 0.5, 0.5, 0, 0.5, 0.5, 0.5, 0.5, 0.5]
        assert training_cases[1].move_index == 3
        assert training_cases[1].target_value == 0.5
        
        assert training_cases[2].input_board == [1, 1, 0.5, 0, 0.5, 0.5, 0.5, 0.5, 0.5]
        assert training_cases[2].move_index == 1
        assert training_cases[2].target_value == 1
        
        assert training_cases[3].input_board == [1, 1, 0.5, 0, 0, 0.5, 0.5, 0.5, 0.5]
        assert training_cases[3].move_index == 4
        assert training_cases[3].target_value == 0.5
        
        assert training_cases[4].input_board == [1, 1, 1, 0, 0, 0.5, 0.5, 0.5, 0.5]
        assert training_cases[4].move_index == 2
        assert training_cases[4].target_value == 1
        

    def get_simple_board(self):
        # X | X | X
        # O | O | 
        #   |   |

        board = Board()
        board.move(0, "X")
        board.move(3, "O")
        
        board.move(1, "X")
        board.move(4, "O")
        
        board.move(2, "X")
    
        return board