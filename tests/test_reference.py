import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('verify_results', ROOT / 'scripts/verify_results.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


class ReferenceTests(unittest.TestCase):
    def test_pose_change_and_incomplete_split_are_detected(self):
        expected = json.loads(reference.REFERENCE.read_text())
        actual = copy.deepcopy(expected)
        self.assertEqual(reference.compare(expected, actual), [])
        actual['frames'][0]['objects'][0]['translation_m'][0] += 0.01
        self.assertTrue(any('translation_m' in e for e in reference.compare(expected, actual)))
        actual['frames'].pop()
        self.assertTrue(any('Frame IDs' in e for e in reference.compare(expected, actual)))


if __name__ == '__main__':
    unittest.main()
