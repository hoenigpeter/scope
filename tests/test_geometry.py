"""Geometry contracts independent of model predictions and learned pose quality."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scope.geometry import backproject, crop, restore, register


class GeometryTests(unittest.TestCase):
    def test_backprojection_units_and_invalid_depth(self):
        depth = np.array([[1, 0], [np.nan, 2.]])
        points, pixels = backproject(depth, [2, 2, 0, 0])
        np.testing.assert_allclose(points, [[0, 0, 1], [1, 1, 2]])
        np.testing.assert_array_equal(pixels[0], [0, 1])

    def test_clipped_crop_restores_pixel_location(self):
        image = np.zeros((60, 80, 3), np.uint8)
        image[0:30, 0:20] = [17, 43, 91]
        cropped, meta = crop(image, [0, 0, 20, 30], size=160)
        restored = restore(cropped, meta, image.shape)
        np.testing.assert_array_equal(restored[5:25, 5:15], image[5:25, 5:15])
        self.assertTrue(np.all(restored[40:] == 0))

    def test_similarity_with_outliers_and_camera_axes(self):
        rng = np.random.default_rng(15)
        nocs = rng.integers(20, 235, (150, 3), dtype=np.uint8)
        src = nocs / 127.5 - 1
        angle = 0.7
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        translation = np.array([0.12, -0.08, 0.8])
        points = 0.2 * (src @ rotation.T) + translation
        points[:15] += rng.normal(0, 0.3, (15, 3))
        r, t, s, quality = register(nocs, points, noise_bound=0.002)
        np.testing.assert_allclose(r, rotation, atol=0.015)
        np.testing.assert_allclose(t, translation, atol=0.003)
        self.assertAlmostEqual(s, 0.2, delta=0.003)
        self.assertGreater(quality['inlier_fraction'], 0.8)


if __name__ == '__main__':
    unittest.main()
