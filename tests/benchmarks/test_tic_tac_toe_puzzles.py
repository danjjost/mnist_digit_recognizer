from decimal import Decimal
import unittest

from src.neuralnet.network import Network
from src.pipeline.evaluation import Evaluation
"""
from src.pipeline.evaluation_epoch import EvaluationEpoch
from src.pipeline.pipeline import Pipeline
from src.pipeline.population_generator import PopulationGenerator
from src.pipeline.population_modifiers.population_evolver import PopulationEvolver
"""
from src.tictactoe.ai_player import AiPlayer
from src.tictactoe.board import Board

class Puzzle:
    def __init__(self, current_state: list[Decimal], desired_move_index: int, desired_move_value: Decimal):
        self.current_state: list[Decimal] = current_state
        self.desired_move_index: int = desired_move_index
        self.desired_move_value: Decimal = desired_move_value

class PuzzleEvaluation(Evaluation):
    def __init__(self, puzzles: list[Puzzle], back_propogate: bool = True):
        self.puzzles = puzzles
        self.back_propogate = back_propogate

    def evaluate(self, network: Network):
        network.score = 0
        
        for puzzle in self.puzzles:            
            ai_player = AiPlayer(' ')
            ai_player.set_network(network)
            move = ai_player.get_move(Board().from_state(puzzle.current_state))
            
            if move == puzzle.desired_move_index:
                network.score += 1
            elif self.back_propogate:
                network.back_propagate_node_index_and_target(puzzle.desired_move_index, puzzle.desired_move_value)
                network.back_propagate_node_index_and_target(move, Decimal(0.5))
                
        network.apply_gradients()

# X always goes first
class TestTicTacToePuzzles(unittest.TestCase):
    def setUp(self) -> None:
        self.x_move_puzzles: list[Puzzle] = [
            Puzzle([ # Top Row
             Decimal(1), Decimal(1), Decimal(0.5), 
             Decimal(0), Decimal(0.5), Decimal(0.5),
             Decimal(0.5), Decimal(0), Decimal(0.5)
            ], 2, Decimal(1)),
            Puzzle([ # Middle Row
                Decimal(0.5), Decimal(0.5), Decimal(0),
                Decimal(1), Decimal(0.5), Decimal(1),
                Decimal(0.5), Decimal(0), Decimal(0.5)
            ], 4, Decimal(1)),
            Puzzle([ # Bottom Row
                Decimal(0.5), Decimal(0), Decimal(0.5),
                Decimal(0.5), Decimal(0), Decimal(0.5),
                Decimal(0.5), Decimal(1), Decimal(1)
            ], 6, Decimal(1)),
            Puzzle([ # Left Column
                Decimal(1), Decimal(0.5), Decimal(0.5),
                Decimal(1), Decimal(0), Decimal(0.5),
                Decimal(0.5), Decimal(0.5), Decimal(0)
            ], 6, Decimal(1)),
            Puzzle([ # Middle Column
                Decimal(0), Decimal(1), Decimal(0),
                Decimal(0.5), Decimal(0.5), Decimal(0.5),
                Decimal(0.5), Decimal(1), Decimal(0.5)
            ], 4, Decimal(1)), 
            Puzzle([ # Right Column
                Decimal(0), Decimal(0.5), Decimal(1),
                Decimal(0), Decimal(0.5), Decimal(1),
                Decimal(0.5), Decimal(0.5), Decimal(0.5)
            ], 8, Decimal(1)),
            Puzzle([ # Top Left to Bottom Right Diagonal
                Decimal(0.5), Decimal(0.5), Decimal(0.5),
                Decimal(0.5), Decimal(1), Decimal(0.5),
                Decimal(0), Decimal(0), Decimal(1)
            ], 0, Decimal(1)),
            Puzzle([ # Top Right to Bottom Left Diagonal
                Decimal(0), Decimal(0), Decimal(0.5),
                Decimal(0.5), Decimal(1), Decimal(0.5),
                Decimal(1), Decimal(0.5), Decimal(0.5)
            ], 2, Decimal(0)),
        ]
        
        self.o_move_puzzles: list[Puzzle] = [
            Puzzle([ # Top Row
             Decimal(0), Decimal(0.5), Decimal(0), 
             Decimal(0.5), Decimal(1), Decimal(0.5),
             Decimal(1), Decimal(1), Decimal(0.5)
            ], 1, Decimal(0)),
            Puzzle([ # Middle Row
                Decimal(1), Decimal(0.5), Decimal(0.5),
                Decimal(0), Decimal(0), Decimal(0.5),
                Decimal(1), Decimal(0.5), Decimal(1)
            ], 5, Decimal(0)),
            Puzzle([ # Bottom Row
                Decimal(0.5), Decimal(1), Decimal(0.5),
                Decimal(1), Decimal(0.5), Decimal(1),
                Decimal(0), Decimal(0), Decimal(0.5)
            ], 7, Decimal(0)),
            Puzzle([ # Left Column
                Decimal(0), Decimal(1), Decimal(0.5),
                Decimal(0), Decimal(1), Decimal(0.5),
                Decimal(0.5), Decimal(0.5), Decimal(1)
            ], 6, Decimal(0)),
            Puzzle([ # Middle Column
                Decimal(1), Decimal(0), Decimal(0.5),
                Decimal(0.5), Decimal(0), Decimal(1),
                Decimal(1), Decimal(0.5), Decimal(0.5)
            ], 7, Decimal(0)), 
            Puzzle([ # Right Column
                Decimal(1), Decimal(1), Decimal(0),
                Decimal(0.5), Decimal(0.5), Decimal(0),
                Decimal(0.5), Decimal(1), Decimal(0.5)
            ], 8, Decimal(0)),
            Puzzle([ # Top Left to Bottom Right Diagonal
                Decimal(0), Decimal(0.5), Decimal(1),
                Decimal(1), Decimal(0), Decimal(0.5),
                Decimal(1), Decimal(0.5), Decimal(0.5)
            ], 8, Decimal(0)),
            Puzzle([ # Top Right to Bottom Left Diagonal
                Decimal(1), Decimal(0.5), Decimal(0.5),
                Decimal(1), Decimal(0), Decimal(0.5),
                Decimal(0), Decimal(0.5), Decimal(1)
            ], 2, Decimal(0)),
        ]
    
    """ 
    def test_can_solve_x_puzzles(self):
        print('Running X puzzles')
        population = PopulationGenerator().generate(10, [9, 18, 18, 18, 9])
        evaluation = PuzzleEvaluation(self.x_move_puzzles)
        
        evaluationEpoch = EvaluationEpoch(evaluation)
        
        pipeline = Pipeline()
        pipeline.add(evaluationEpoch)
        pipeline.add(PopulationEvolver(0.3))
        
        for _ in range(100):
            population = pipeline.run(population)
        
        assert population.average_score == 8
        assert population.max_score == 8
        assert population.min_score == 8 
    """