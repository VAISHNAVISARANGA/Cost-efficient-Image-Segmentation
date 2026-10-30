import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import compute_ces          # imports torch via utils

class T(unittest.TestCase):
    def test_zero_if_slow(self): self.assertEqual(compute_ces(0.9, 50.01, 1000), 0.0)
    def test_formula(self): self.assertAlmostEqual(compute_ces(0.8, 25.0, 2 * 1024 * 1024), 0.8 * (1 - 0.1 - 0.05))
    def test_size_cap(self): self.assertAlmostEqual(compute_ces(1.0, 0.0, 100 * 1024 * 1024), 0.9)

if __name__ == "__main__":
    unittest.main()
