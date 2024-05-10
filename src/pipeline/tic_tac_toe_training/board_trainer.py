from src.neuralnet.network import Network
from src.pipeline.tic_tac_toe_training.board_evaluator import BoardEvaluator
from src.tictactoe.board import Board


class BoardTrainer():
    def __init__(self, board_evaluator: BoardEvaluator):
        self.board_evaluator = board_evaluator
    
    def train(self, board: Board, network: Network):
        training_cases = self.board_evaluator.get_training_cases(board)
        
        for training_case in training_cases:
            network.set_input(training_case.input_board)
            network.feed_forward()
            network.back_propagate_node_index_and_target(training_case.move_index, training_case.target_value)
            
        network.apply_gradients()