"""Run from project root: python -m unittest discover -s tests -v   (test_metrics needs only numpy)"""
import os, sys, unittest
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics import boundary, boundary_f1, confusion_2x2, miou_from_cm, quality, MetricAccumulator


def square(n=32, y0=8, x0=8, s=12):
    m = np.zeros((n, n), np.uint8); m[y0:y0 + s, x0:x0 + s] = 1; return m


def brute_bf1(pred, gt, r=2):
    """Slow literal implementation of the assignment text, used to cross-check the vectorised one."""
    def bnd(m):
        h, w = m.shape; pts = []
        for y in range(h):
            for x in range(w):
                if m[y, x]:
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        yy, xx = y + dy, x + dx
                        if not (0 <= yy < h and 0 <= xx < w) or not m[yy, xx]:
                            pts.append((y, x)); break
        return pts
    pb, gb = bnd(pred), bnd(gt)
    if not pb and not gb: return 1.0
    if not pb or not gb: return 0.0
    near = lambda p, S: any(abs(p[0] - q[0]) <= r and abs(p[1] - q[1]) <= r for q in S)
    P = sum(near(p, gb) for p in pb) / len(pb); R = sum(near(g, pb) for g in gb) / len(gb)
    return 0.0 if P + R == 0 else 2 * P * R / (P + R)


class T(unittest.TestCase):
    def test_identical(self):
        m = square(); cm = confusion_2x2(m, m)
        self.assertEqual(miou_from_cm(cm)[0], 1.0); self.assertEqual(boundary_f1(m, m), 1.0)

    def test_no_boundaries_both(self):
        z = np.zeros((16, 16), np.uint8); self.assertEqual(boundary_f1(z, z), 1.0)

    def test_empty_prediction(self):
        self.assertEqual(boundary_f1(np.zeros((32, 32), np.uint8), square()), 0.0)
        cm = confusion_2x2(np.zeros((32, 32), np.uint8), square()); miou, (ib, i_f) = miou_from_cm(cm)
        self.assertEqual(i_f, 0.0); self.assertAlmostEqual(miou, ib / 2)

    def test_image_border_counts_as_background(self):
        self.assertEqual(int(boundary(np.ones((8, 8), np.uint8)).sum()), 28)   # outer ring of 8x8

    def test_shift_within_tolerance(self):
        self.assertEqual(boundary_f1(square(x0=10), square()), 1.0)            # 2 px shift is inside 5x5

    def test_shift_outside_tolerance(self):
        self.assertLess(boundary_f1(square(x0=13), square()), 1.0)             # 5 px shift

    def test_matches_bruteforce(self):
        rng = np.random.default_rng(1)
        for _ in range(25):
            a = (rng.random((14, 14)) > 0.5).astype(np.uint8); b = (rng.random((14, 14)) > 0.5).astype(np.uint8)
            self.assertAlmostEqual(boundary_f1(a, b), brute_bf1(a, b), places=9)
            s1 = square(14, 3, 3, 6); s2 = np.roll(s1, int(rng.integers(-4, 5)), axis=1)
            self.assertAlmostEqual(boundary_f1(s2, s1), brute_bf1(s2, s1), places=9)

    def test_miou_accumulated_not_per_image(self):
        acc = MetricAccumulator(); a, b = square(), square(s=4)
        acc.update(a, a); acc.update(np.zeros_like(b), b)
        cm = confusion_2x2(a, a) + confusion_2x2(np.zeros_like(b), b)
        self.assertAlmostEqual(acc.summary()["mIoU"], miou_from_cm(cm)[0])

    def test_quality(self):
        self.assertAlmostEqual(quality(0.8, 0.5), 0.71)


if __name__ == "__main__":
    unittest.main()
