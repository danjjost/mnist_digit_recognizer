from random import Random
import unittest

from src.digit_recognition.convolution.image_convolver import ImageConvolver


class TestIntegratedImageConvolver(unittest.TestCase):
    def test_returns_appropriately_sized_array_after_convolution(self):
        input_image = self.generate_random_image(28)
        
        output = ImageConvolver().convolve(input_image)
        
        assert len(output) == 1352
        
    def generate_random_image(self, length: int):
        return [Random().random() for _ in range(length ** 2)]