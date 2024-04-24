import unittest
from honors1 import Wave, SAMPLERATE as RATE
import numpy as np

class TestWaveOperations(unittest.TestCase):
    def setUp(self):
        # Initialize some waves with known properties
        self.wave1 = Wave(frequency=440, duration=1.0)  # A4 note, 1 second
        self.wave2 = Wave(frequency=440, duration=1.0)  # Same as wave1
        self.wave3 = Wave(frequency=880, duration=0.5)  # A5 note, 0.5 seconds

    def test_addition(self):
        # Test addition of wave1 and wave2
        result_wave = self.wave1 + self.wave2
        expected_data = self.wave1.data + self.wave2.data
        self.assertTrue(np.allclose(result_wave.data, expected_data), "Addition does not match expected result")

    def test_subtraction(self):
        # Test subtraction of wave1 from itself
        result_wave = self.wave1 - self.wave1
        expected_data = np.zeros_like(self.wave1.data)
        self.assertTrue(np.allclose(result_wave.data, expected_data), "Subtraction does not match expected result")

    def test_multiplication(self):
        # Test multiplication of wave1 and wave3
        result_wave = self.wave1 * self.wave3
        # Create expected data with zeros to match the shorter duration of wave3
        min_length = min(len(self.wave1.data), len(self.wave3.data))
        expected_data = np.zeros_like(self.wave1.data)
        expected_data[:min_length] = self.wave1.data[:min_length] * self.wave3.data[:min_length]
        self.assertTrue(np.allclose(result_wave.data, expected_data), "Multiplication does not match expected result")

if __name__ == '__main__':
    unittest.main()
