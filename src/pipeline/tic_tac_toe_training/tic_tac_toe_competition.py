from src.neuralnet.network import Network
from src.pipeline.tic_tac_toe_training.board_trainer import BoardTrainer
from src.tictactoe.ai_player import AiPlayer
from src.tictactoe.game import Game
from src.pipeline.competition import Competition


class TicTacToeCompetition(Competition):
    def __init__(self, game: Game, board_trainer: BoardTrainer, x_player: AiPlayer, o_player: AiPlayer):
        super().__init__()
        self.game = game
        self.board_trainer = board_trainer
        self.x_player: AiPlayer = x_player
        self.o_player: AiPlayer = o_player
        
    def compete(self, challenger: Network, challenged: Network):
        self.x_player.set_network(challenger)
        self.o_player.set_network(challenged)
                
        self.game.play(self.x_player, self.o_player)

        self.board_trainer.train(self.game.board, challenger)
        self.board_trainer.train(self.game.board, challenged)