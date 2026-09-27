import tempfile
from pathlib import Path
import struct
import unittest

import numpy as np

from convert_ft_factorizer_off import convert_state, index_records


class FoldTest(unittest.TestCase):
    def test_fold_and_keep_base_moments(self):
        with tempfile.TemporaryDirectory() as folder:
            src, dst = Path(folder) / 'source', Path(folder) / 'dest'
            values = np.arange(12, dtype='<f4')
            records = {'nnue/train/shared_coefficients': np.array([0.5, 1], dtype='<f4'),
                       'nnue/train/completed_steps': np.array([100], dtype='<f4')}
            for section in ('weights', 'slow', 'momentum', 'velocity'):
                records[f'nnue/{section}/l0w'] = values
            with src.open('wb') as f:
                for name, v in records.items():
                    f.write(name.encode() + b'\n' + struct.pack('<Q', len(v)) + v.tobytes())
            original = src.read_bytes()
            self.assertEqual(convert_state(src, dst, 4, 2, 2), 0.5)
            index = index_records(dst)
            with dst.open('rb') as f:
                for section in ('weights', 'slow', 'momentum', 'velocity'):
                    offset, count = index[f'nnue/{section}/l0w']
                    self.assertEqual(count, 8)
                    f.seek(offset)
                    actual = np.frombuffer(f.read(count * 4), dtype='<f4')
                    expected = values[:8].copy()
                    if section in ('weights', 'slow'):
                        expected += np.tile(values[8:] * 0.5, 2)
                    np.testing.assert_array_equal(actual, expected)
            self.assertEqual(src.read_bytes(), original)
            with self.assertRaises(FileExistsError):
                convert_state(src, dst, 4, 2, 2)

    def test_truncated_input_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'bad'
            p.write_bytes(b'weights\n' + struct.pack('<Q', 10))
            with self.assertRaises(ValueError):
                index_records(p)
