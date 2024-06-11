import unittest
from unittest.mock import MagicMock

from src.neuralnet.network import Network
from src.pipeline.evaluation import Evaluation
from src.pipeline.evaluation_epoch import EvaluationEpoch
from src.pipeline.population import PopulationDTO


class TestEvaluationEpoch(unittest.TestCase):
    def test_epoch_runs_evaluation_for_each_network(self):
        networks = [Network([1, 1]), Network([1, 1])]
        populationDto = PopulationDTO(networks)
        
        evaluation = MagicMock(spec=Evaluation)
        
        evaluationEpoch = EvaluationEpoch(evaluation)
        
        evaluationEpoch.run(populationDto)
        
        evaluation.evaluate.assert_any_call(networks[0])
        evaluation.evaluate.assert_any_call(networks[1])