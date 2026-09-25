import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location('prepare_data', Path(__file__).resolve().parents[1] / 'scripts/prepare_data.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class DataTests(unittest.TestCase):
    def test_coco_rle_is_column_major_foreground(self):
        mask = prepare.decode_rle({'counts': [1, 2, 2, 1], 'size': [2, 3]}, 2, 3)
        np.testing.assert_array_equal(mask, [[0, 1, 0], [1, 0, 1]])

    def test_malformed_rle_is_rejected(self):
        with self.assertRaises(ValueError):
            prepare.decode_rle({'counts': [1, 1], 'size': [2, 3]}, 2, 3)


if __name__ == '__main__':
    unittest.main()
