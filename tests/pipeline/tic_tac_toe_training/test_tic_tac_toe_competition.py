import unittest
from unittest.mock import MagicMock, Mock, patch

from src.neuralnet.network import Network
from src.pipeline.competition import Competition
from src.pipeline.tic_tac_toe_training.board_trainer import BoardTrainer
from src.pipeline.tic_tac_toe_training.tic_tac_toe_competition import TicTacToeCompetition
from src.tictactoe.ai_player import AiPlayer
from src.tictactoe.board import Board
from src.tictactoe.game import Game

