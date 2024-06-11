from decimal import Decimal


class Board():
        
    def __init__(self):
        self.winner = ''
        self.current = [
            "", "", "", 
            "", "", "", 
            "", "", ""
        ]
        self.history: list[list[str]] = []
    
    def move(self, position_index: int, symbol: str):
        if(self.is_valid_move(position_index) == False):
            raise ValueError("Invalid move")
            
        self.current[position_index] = symbol
        self.history.append(self.current.copy())
    
    
    def is_valid_move(self, position_index: int) -> bool:
        return position_index >= 0 and position_index < 9 and self.current[position_index] == ""
    
    
    def is_board_full(self) -> bool:
        return "" not in self.current
    
    
    def get_winner(self) -> str:
        winning_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # columns
            [0, 4, 8], [2, 4, 6]  # diagonals
        ]

        for combination in winning_combinations:
            space_1_symbol = self.current[combination[0]]
            space_2_symbol = self.current[combination[1]]
            space_3_symbol = self.current[combination[2]]
            
            if space_1_symbol != "" and space_1_symbol == space_2_symbol == space_3_symbol:
                self.winner = space_1_symbol
                break  # Exit the loop once a winner is found

        return self.winner
    
    
    def is_game_over(self) -> bool:
        return self.get_winner() != '' or self.is_board_full()
    
    
    def get_current_digits(self) -> list[Decimal]:
        return [
            Decimal(0.5) if square == ''
            else Decimal(1) if square == 'X' 
            else Decimal(0) for square in self.current
        ]
         
    def from_state(self, digits: list[Decimal]) -> 'Board':
        self.current = [
            'X' if digit == Decimal(1)
            else 'O' if digit == Decimal(0)
            else '' for digit in digits
        ]
        
        return self