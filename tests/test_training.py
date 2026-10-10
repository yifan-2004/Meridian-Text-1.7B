"""Leakage and malformed-input regression checks, no model libraries needed."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from train import load_splits, read_split


class TrainingInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def write(self, name, records):
        path = self.root / name
        path.write_text(''.join(json.dumps(row) + '\n' for row in records))
        return path

    def test_valid_split(self):
        train = self.write('train', [{'document_id': 'a', 'text': 'Alpha.'}])
        val = self.write('val', [{'document_id': 'b', 'text': 'Beta.'}])
        self.assertEqual(load_splits(train, val), (['Alpha.'], ['Beta.']))

    def test_document_leakage(self):
        train = self.write('train', [{'document_id': 'a', 'text': 'Alpha.'}])
        val = self.write('val', [{'document_id': 'a', 'text': 'Different chunk.'}])
        with self.assertRaisesRegex(ValueError, 'overlap'):
            load_splits(train, val)

    def test_text_leakage_normalizes_whitespace(self):
        train = self.write('train', [{'document_id': 'a', 'text': 'Alpha beta'}])
        val = self.write('val', [{'document_id': 'b', 'text': ' Alpha  beta\n'}])
        with self.assertRaisesRegex(ValueError, 'overlap'):
            load_splits(train, val)

    def test_duplicate_document_rejected(self):
        path = self.write('train', [{'document_id': 'a', 'text': 'Alpha'}, {'document_id': 'a', 'text': 'Beta'}])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            read_split(path)

    def test_missing_text_rejected(self):
        with self.assertRaisesRegex(ValueError, 'required'):
            read_split(self.write('train', [{'document_id': 'a'}]))

    def test_empty_split_rejected(self):
        with self.assertRaisesRegex(ValueError, 'empty'):
            read_split(self.write('train', []))


if __name__ == '__main__':
    unittest.main()
