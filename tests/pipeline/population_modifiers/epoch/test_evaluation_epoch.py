import unittest
from unittest.mock import MagicMock

from src.neuralnet.network import Network
from src.pipeline.population_modifiers.epoch.i_evaluation import IEvaluation
from src.pipeline.population import PopulationDTO
from src.pipeline.population_modifiers.epoch.evaluation_epoch import EvaluationEpoch


class TestEvaluationEpoch(unittest.TestCase):
    def test_epoch_runs_evaluation_for_each_network(self):
        networks = [Network([1, 1]), Network([1, 1])]
        populationDto = PopulationDTO(networks)
        
        evaluation = MagicMock(spec=IEvaluation)
        
        evaluationEpoch = EvaluationEpoch(evaluation)
        
        evaluationEpoch.run(populationDto)
        
        evaluation.evaluate.assert_any_call(networks[0])
        evaluation.evaluate.assert_any_call(networks[1])