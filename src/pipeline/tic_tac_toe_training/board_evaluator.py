
from src.pipeline.tic_tac_toe_training.board_training_case import BoardTrainingCase
from src.tictactoe.board import Board


class BoardEvaluator():
    def get_training_cases(self, board: Board) -> list[BoardTrainingCase]:
        history = board.history
        winner = board.get_winner()
        winning_digit = self.get_digit(winner)
        
        if(winning_digit == float(0.5)):
            return []
        
        training_cases: list[BoardTrainingCase] = []
        
        for i in range(len(history)):
            current_board = history[i]
            
            previous_board: list[str] = ["", "", "", "", "", "", "", "", ""]
            
            if i != 0:
                previous_board = history[i - 1]
                 
            move_index = self.get_move_index(current_board, previous_board)
            move_target = self.get_move_target(current_board, move_index, winning_digit)
            
            training_case = BoardTrainingCase(self.get_board_digits(current_board), move_index, move_target)
            
            training_cases.append(training_case)
        
        return training_cases
            
    
    def get_move_index(self, current_board: list[str], previous_board: list[str]) -> int:
        for j in range(len(current_board)):
            if current_board[j] != previous_board[j]:
                return j
        
        raise ValueError("No move found")
        
    def get_move_target(self, current_board: list[str], move_index: int, winning_digit: float) -> float:
        if self.get_digit(current_board[move_index]) == winning_digit:
            return winning_digit
        else:
            return float(0.5)
    
    def get_digit(self, symbol: str) -> float:
        if symbol == "":
            return float(0.5)
        elif symbol == "X":
            return float(1)
        else:
            return float(0)
        
    def get_board_digits(self, string_board: list[str]) -> list[float]:
        return [self.get_digit(square) for square in string_board]