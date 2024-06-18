import unittest
from unittest.mock import MagicMock

from config import Config, NetworkEvaluationMode
from src.digit_recognition.image_loader import ImageLoader
from src.digit_recognition.mnist_evaluation import MNISTEvaluation
from src.digit_recognition.mnist_image_evaluator import MNISTImageEvaluator
from src.neuralnet.network import Network


class TestMNISTEvaluation(unittest.TestCase):

    def setUp(self) -> None:
        self.initializeConfig()
        self.mnist_image_evaluator = MagicMock(spec=MNISTImageEvaluator)
        self.batch_loader = MagicMock(spec=ImageLoader)
        self.mnist_evaluation = MNISTEvaluation(self.config, self.batch_loader, self.mnist_image_evaluator)


    def initializeConfig(self):
        self.BATCH_SIZE = 15
        self.TRAINING_MODE = NetworkEvaluationMode.TRAIN
        
        self.config = MagicMock(spec=Config)
        self.config.mode = self.TRAINING_MODE
        self.config.training_batch_size = self.BATCH_SIZE

    
    def test_evaluate_retrieves_batch_of_configured_size(self):
        network = Network([1,1])
        self.config.training_batch_size = 10
        
        
        self.mnist_evaluation.evaluate(network)
        
        
        self.assertEqual(self.batch_loader.get_training_image.call_count, 10)
        
        
    def test_evaluate_retrieves_testing_batch_when_configured(self):
        network = Network([1,1])
        self.config.mode = NetworkEvaluationMode.TEST
        self.config.training_batch_size = 10
        
        self.mnist_evaluation.evaluate(network)
        
        self.assertEqual(self.batch_loader.get_testing_image.call_count, 10)
        
    def test_evaluate_calls_image_evaluator_with_each_image_in_batch(self):
        network = Network([1,1])
        mock_image = MagicMock()
        self.batch_loader.get_training_image.return_value = mock_image
        
        
        self.mnist_evaluation.evaluate(network)
        
        
        self.mnist_image_evaluator.evaluate_image.assert_any_call(network, mock_image)