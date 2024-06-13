from src.neuralnet.network import Network
from src.tictactoe.board import Board
from src.tictactoe.player import Player


class AiPlayer(Player):
    def __init__(self, symbol: str):
        super().__init__(symbol)
        self.network: Network | None = None
            
    def set_network(self, network: Network):
        self.network = network
            
    def get_move(self, board: Board) -> int:
        self.try_get_network()
        
        move_probabilities = self.get_move_probabilities(board)
        
        move_selection_order = self.get_move_selection_order(move_probabilities)

        for move_index in move_selection_order:
            if board.is_valid_move(move_index):
                return move_index
            
        raise ValueError("No valid moves found")
    
    
    def get_move_selection_order(self, move_probabilities: list[float]) -> list[int]:
        # Create an enumerated list of tuples, where each tuple is (index, probability)
        indexed_probs: list[tuple[int, float]] = list(enumerate(move_probabilities))

        sort_order_descending = True
        
        if self.get_digit() == float(0):
            sort_order_descending = False
        
        # Sort the list of tuples based on the probability, in descending order
        # The key argument determines the sorting criteria, and we use a lambda function
        # to sort by probability (x[1]), and `reverse=True` to get descending order
        sorted_by_probability = sorted(indexed_probs, key=lambda x: x[1], reverse=sort_order_descending)

        # Extract the indices from the sorted list of tuples
        move_selection_order = [index for index, _ in sorted_by_probability]

        return move_selection_order
        
        
    def get_valid_move_indexes(self, board: Board) -> list[int]:
        valid_moves: list[int] = []
        
        for i in range(len(board.current)):
            if board.is_valid_move(i):
                valid_moves.append(i)
        
        return valid_moves


    def try_get_network(self) -> Network:
        if self.network == None:
            raise ValueError("Network not set")

        return self.network


    def get_move_probabilities(self, board: Board) -> list[float]:
        network = self.try_get_network()
        
        network.set_input(board.get_current_digits())

        network.feed_forward()

        return network.get_outputs()
    
    
    def get_digit(self):
        if self.symbol == "":
            return 0.5
        elif self.symbol == "X":
            return 1
        else:
            return 0