import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('fetch_assets', Path(__file__).resolve().parents[1] / 'scripts/fetch_assets.py')
assets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assets)


class AssetTests(unittest.TestCase):
    def bundle(self, root, contents, member='weights/scope.pth', file_hash=None):
        archive = root / 'bundle.tar.gz'
        with tarfile.open(archive, 'w:gz') as tar:
            info = tarfile.TarInfo(member)
            info.size = len(contents)
            tar.addfile(info, io.BytesIO(contents))
        entry = dict(filename=archive.name, url='https://example.invalid/bundle.tar.gz',
                     bytes=archive.stat().st_size, sha256=assets.digest(archive),
                     files={member: file_hash or hashlib.sha256(contents).hexdigest()})
        return archive, entry

    def test_verified_install_and_offline_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive, entry = self.bundle(root, b'checkpoint')
            assets.install_archive(root, entry, archive, root/'staging')
            self.assertTrue(assets.installed(root, entry))
            (root/'assets.json').write_text(json.dumps({'assets': [entry]}))
            with patch.object(assets, 'download', side_effect=AssertionError('Unexpected network access')):
                assets.ensure_assets(root)
            (root/'weights/scope.pth').write_bytes(b'corrupt')
            self.assertFalse(assets.installed(root, entry))

    def test_bad_file_checksum_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'weights').mkdir()
            (root/'weights/scope.pth').write_bytes(b'old')
            archive, entry = self.bundle(root, b'new', file_hash='0'*64)
            with self.assertRaisesRegex(ValueError, 'File checksum mismatch'):
                assets.install_archive(root, entry, archive, root/'staging')
            self.assertEqual((root/'weights/scope.pth').read_bytes(), b'old')

    def test_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive, entry = self.bundle(root, b'bad', member='data/real275_demo/../../setup.sh')
            with self.assertRaisesRegex(ValueError, 'Invalid asset path'):
                assets.install_archive(root, entry, archive, root/'staging')

    def test_corrupt_archive_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive, entry = self.bundle(root, b'checkpoint')
            archive.write_bytes(b'broken')
            with self.assertRaisesRegex(ValueError, 'Archive checksum/size mismatch'):
                assets.install_archive(root, entry, archive, root/'staging')


if __name__ == '__main__':
    unittest.main()
