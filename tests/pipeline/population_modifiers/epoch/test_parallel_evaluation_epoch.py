import unittest
from unittest.mock import MagicMock

from src.neuralnet.network import Network
from src.pipeline.population_modifiers.epoch.evaluation import Evaluation
from src.pipeline.population_modifiers.epoch.parallel_evaluation_epoch import ParallelEvaluationEpoch
from src.pipeline.population import PopulationDTO


class TestParallelEvaluationEpoch(unittest.TestCase):
    def test_epoch_runs_evaluation_for_each_network(self):
        networks = [Network([1, 1]), Network([1, 1])]
        populationDto = PopulationDTO(networks)
        
        evaluation = MagicMock(spec=Evaluation)
        
        evaluationEpoch = ParallelEvaluationEpoch(evaluation)
        
        evaluationEpoch.run(populationDto)
        
        evaluation.evaluate.assert_any_call(networks[0])
        evaluation.evaluate.assert_any_call(networks[1])