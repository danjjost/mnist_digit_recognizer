from decimal import Decimal
import unittest

from src.tictactoe.board import Board


class TestBoard(unittest.TestCase):
    def test_move(self):
        board = Board()
        
        board.move(0, 'X')
        
        self.assertEqual(
            board.current, 
            ['X', '', '', 
            '', '', '',
            '', '', '']
        )
     
    def test_invalid_move(self):
        board = Board()
        
        board.move(0, 'X')
        
        with self.assertRaises(ValueError):
            board.move(0, 'X')
            
    def test_is_board_full(self):
        board = Board()
        
        self.assertFalse(board.is_board_full())
    
        board.move(0, 'X')
        board.move(1, 'O')
        board.move(2, 'X')
        board.move(3, 'O')
        board.move(4, 'X')
        board.move(5, 'O')
        board.move(6, 'X')
        board.move(7, 'O')
        board.move(8, 'X')
        
        self.assertTrue(board.is_board_full())
        
    def test_get_winner(self):
        board = Board()
        
        board.move(0, 'X')
        board.move(1, 'X')
        board.move(2, 'X')
        
        self.assertEqual(board.get_winner(), 'X')
        
    def test_board_history(self):
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
        
        self.assertEqual(
            board.history, 
            [
                ['X', '', '', 
                '', '', '',
                '', '', ''],
                ['X', 'O', '', 
                '', '', '',
                '', '', ''],
                ['X', 'O', 'X', 
                '', '', '',
                '', '', ''],
                ['X', 'O', 'X', 
                'O', '', '',
                '', '', ''],
                ['X', 'O', 'X', 
                'O', 'X', '',
                '', '', ''],
                ['X', 'O', 'X', 
                'O', 'X', 'O',
                '', '', ''],
                ['X', 'O', 'X', 
                'O', 'X', 'O',
                'X', '', ''],
                ['X', 'O', 'X', 
                'O', 'X', 'O',
                'X', 'O', ''],
                ['X', 'O', 'X', 
                'O', 'X', 'O',
                'X', 'O', 'X']
            ]
        )
        
    def test_board_get_digits(self):
        board = Board()
        board.move(0, 'X')
        board.move(1, 'O')
        
        self.assertEqual(
            board.get_current_digits(),
            [
                Decimal(1), Decimal(0), Decimal(0.5), 
                Decimal(0.5), Decimal(0.5), Decimal(0.5), 
                Decimal(0.5), Decimal(0.5), Decimal(0.5)
            ]
        )
        
    def test_get_winner_returns_empty_string_if_none(self):        
        board = Board()
        
        board.move(0, 'X')
        board.move(6, 'O')
        board.move(5, 'X')
        board.move(1, 'O')
        board.move(7, 'X')
        board.move(4, 'O')
        board.move(2, 'X')
        board.move(8, 'O')
        board.move(3, 'X')
        
        
        self.assertEqual(board.get_winner(), '')