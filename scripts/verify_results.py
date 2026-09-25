#!/usr/bin/env python3
"""Compare the fixed REAL275 demo subset with the released reference poses."""
import argparse
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'docs/real275-reference.json'


def compare(reference, actual):
    errors = []
    for key in ['seed', 'steps', 'min_score', 'checkpoint_sha256', 'manifest_sha256']:
        if reference[key] != actual.get(key):
            errors.append(f'{key} differs from the reference')
    if actual.get('failures') != 0:
        errors.append('Run contains failed registrations')
    expected_frames = {f['id']: f for f in reference['frames']}
    actual_frames = {f['id']: f for f in actual['frames']}
    if len(actual_frames) != len(actual['frames']) or expected_frames.keys() != actual_frames.keys():
        return errors + ['Frame IDs differ from the complete reference subset']
    for frame_id, expected in expected_frames.items():
        observed = actual_frames[frame_id]
        expected_objects = {o['id']: o for o in expected['objects']}
        actual_objects = {o['id']: o for o in observed['objects']}
        if len(actual_objects) != len(observed['objects']) or expected_objects.keys() != actual_objects.keys():
            errors.append(f'{frame_id}: detection IDs differ')
            continue
        for object_id, obj in expected_objects.items():
            other = actual_objects[object_id]
            label = f'{frame_id}/object_{object_id}'
            if 'error' in other:
                errors.append(f'{label}: registration failed')
                continue
            for key in ['category', 'seed']:
                if obj[key] != other.get(key):
                    errors.append(f'{label}: {key} differs')
            for key in ['rotation', 'translation_m', 'scale_m_per_signed_nocs_unit']:
                expected_value = np.asarray(obj[key])
                actual_value = np.asarray(other.get(key, np.nan))
                if expected_value.shape != actual_value.shape or not np.allclose(expected_value, actual_value, atol=1e-4, rtol=0):
                    errors.append(f'{label}: {key} exceeds absolute tolerance 1e-4')
    return errors


def verify(run_path, reference_path=REFERENCE):
    reference = json.loads(reference_path.read_text())
    actual = json.loads(run_path.read_text())
    errors = compare(reference, actual)
    if errors:
        raise ValueError('Reference comparison failed:\n' + '\n'.join(errors[:20]))
    count = sum(len(frame['objects']) for frame in reference['frames'])
    print(f"Reference match: {len(reference['frames'])} REAL275 frames, {count} poses (absolute tolerance 1e-4).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    try:
        verify(args.run)
    except ValueError as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
