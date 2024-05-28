from decimal import Decimal

class BoardTrainingCase():
    def __init__(self, input_board: list[Decimal], move_index: int, target_value: Decimal):
        self.input_board = input_board
        self.move_index = move_index
        self.target_value = target_value


    def get_digit(self, symbol: str) -> float:
        if symbol == "":
            return 0.5
        elif symbol == "X":
            return 1
        else:
            return 0