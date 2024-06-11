from src.file.file_loader import FileLoader
from src.file.file_writer import FileWriter
from src.neuralnet.network_to_dict import NetworkToDict
from src.pipeline.competition_epoch import CompetitionEpoch
from src.pipeline.load_population import LoadPopulation
from src.pipeline.pipeline import Pipeline
from src.pipeline.population import PopulationDTO
from src.pipeline.save_population import SavePopulation
from src.pipeline.tic_tac_toe_training.board_evaluator import BoardEvaluator
from src.pipeline.tic_tac_toe_training.board_trainer import BoardTrainer
from src.pipeline.tic_tac_toe_training.tic_tac_toe_competition import TicTacToeCompetition
from src.tictactoe.ai_player import AiPlayer
from src.tictactoe.game import Game

if __name__ == "__main__":  
    load_path = './populations/population_input.json'
    save_path = './populations/population_output.json'
    network_to_dict = NetworkToDict()
    file_loader = FileLoader()
    file_writer = FileWriter()

    game = Game()

    board_evaluator = BoardEvaluator()
    board_trainer = BoardTrainer(board_evaluator)

    x_player = AiPlayer('X')
    o_player = AiPlayer('O')
    competition = TicTacToeCompetition(game, board_trainer, x_player, o_player)

    # end dependencies


    # population modifiers

    load_population = LoadPopulation(load_path, network_to_dict, file_loader)

    epoch = CompetitionEpoch(competition)

    save_population = SavePopulation(save_path, network_to_dict, file_writer)

    # end modifiers


    pipeline = Pipeline()

    pipeline.add(load_population)
    
    for i in range(100):
        pipeline.add(epoch)
        pipeline.add(save_population)
    
    pipeline.run(PopulationDTO([]))