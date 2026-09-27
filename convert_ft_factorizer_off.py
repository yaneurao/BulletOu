"""Fold SFNN HalfKA2 FT factorizer into a new, float32 checkpoint.

Requires numpy. Never overwrites a destination. Individual Ranger moments and
all step counters are retained; virtual-row moments are discarded. Lookahead
slow weights are folded separately. This is not an equivalent optimizer update.
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import struct

import numpy as np


def index_records(path):
    records = {}
    size = path.stat().st_size
    with path.open('rb') as f:
        while f.tell() < size:
            name = f.readline(4096)
            if not name.endswith(b'\n'):
                raise ValueError('Invalid record name')
            name = name[:-1].decode('utf-8')
            length = f.read(8)
            if len(length) != 8 or name in records:
                raise ValueError('Invalid/duplicate record')
            n, = struct.unpack('<Q', length)
            if f.tell() + n * 4 > size:
                raise ValueError('Truncated record')
            records[name] = (f.tell(), n)
            f.seek(n * 4, 1)
    return records


def fold_rows(values, base, virtual, alpha):
    # Same row mapping and f32 operations as fold_sfnn_halfka2_piece_factorized_l0w.
    for start in range(0, base, virtual):
        end = min(start + virtual, base)
        yield np.asarray(values[start:end]) + np.float32(alpha) * values[base:base + end-start]


def convert_state(source, target, base, virtual, width):
    records = index_records(source)
    expected = (base + virtual) * width
    for section in ('weights', 'momentum', 'velocity', 'slow'):
        if records.get(f'nnue/{section}/l0w', (0, -1))[1] != expected:
            raise ValueError(f'Expected factorized FT {section} with {expected} floats')
    coeff_key = 'nnue/train/shared_coefficients'
    if coeff_key not in records or records[coeff_key][1] != 2:
        raise ValueError('Explicit saved FT/shared coefficients required')
    with source.open('rb') as f:
        f.seek(records[coeff_key][0])
        coeff = np.frombuffer(f.read(8), dtype='<f4').copy()
    alpha = float(coeff[0])
    if not np.isfinite(coeff).all() or alpha < 0:
        raise ValueError('Invalid shared coefficient')
    if any(k.startswith('nnue/weights/bn_') for k in records):
        raise ValueError('This converter supports non-BN checkpoints only')
    with source.open('rb') as src, target.open('xb') as dst:
        for name, (offset, n) in records.items():
            changed = name in {f'nnue/{s}/l0w' for s in ('weights', 'slow', 'momentum', 'velocity')}
            dst.write(name.encode() + b'\n')
            dst.write(struct.pack('<Q', base * width if changed else n))
            if name in ('nnue/weights/l0w', 'nnue/slow/l0w'):
                values = np.memmap(source, mode='r', dtype='<f4', offset=offset,
                                   shape=(base + virtual, width))
                for block in fold_rows(values, base, virtual, alpha):
                    if not np.isfinite(block).all():
                        raise ValueError('Non-finite folded weights')
                    dst.write(block.astype('<f4', copy=False).tobytes())
                del values
            elif name == coeff_key:
                coeff[0] = 1.0
                dst.write(coeff.tobytes())
            else:
                src.seek(offset)
                remaining = (base * width if changed else n) * 4
                while remaining:
                    block = src.read(min(4 * 1024 * 1024, remaining))
                    if not block:
                        raise ValueError('Unexpected EOF')
                    dst.write(block)
                    remaining -= len(block)
    # Full verification of master/slow fold, moments, and every untouched record.
    result = index_records(target)
    for name, (offset, n) in records.items():
        out_offset, out_n = result[name]
        a = np.memmap(source, mode='r', dtype='<f4', offset=offset, shape=(n,))
        b = np.memmap(target, mode='r', dtype='<f4', offset=out_offset, shape=(out_n,))
        if name in ('nnue/weights/l0w', 'nnue/slow/l0w'):
            cursor = 0
            for block in fold_rows(a.reshape(base + virtual, width), base, virtual, alpha):
                flat = block.ravel()
                np.testing.assert_array_equal(b[cursor:cursor + flat.size], flat)
                cursor += flat.size
        elif name == coeff_key:
            np.testing.assert_array_equal(b, [1.0, a[1]])
        else:
            for start in range(0, out_n, 1024 * 1024):
                np.testing.assert_array_equal(b[start:start+1024*1024], a[start:min(start+1024*1024, out_n)])
        del a, b
    return alpha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    source, output = args.checkpoint.resolve(), args.output.resolve()
    settings = json.loads((source / 'bulletou-settings.json').read_text(encoding='utf-8-sig'))
    match = re.fullmatch(r'SFNN_halfka2_(\d+)_\d+_\d+(?:_.*)?', settings['arch'])
    if not match:
        raise ValueError('Only SFNN_halfka2 supported')
    width = int(match[1])
    output.mkdir(parents=True, exist_ok=False)
    temporary = output / 'state.bin.incomplete'
    alpha = convert_state(source / 'state.bin', temporary, 81 * 1629, 1629, width)
    temporary.rename(output / 'state.bin')
    for name in ('dataloader_pos.txt', 'progress.bin', 'teacher.txt'):
        if (source / name).exists():
            shutil.copy2(source / name, output / name)
    for key in ('sfnn_ft_factorizer', 'no_ft_factorize'):
        settings.pop(key, None)
    settings['ft_factorizer'] = False
    settings['ft_factorizer_alpha'] = 1.0
    (output / 'bulletou-settings.json').write_text(json.dumps(settings, indent=2) + '\n', encoding='utf-8')
    report = dict(source=str(source), alpha=alpha, verification='all records checked',
                  optimizer='base FT moments retained; virtual moments discarded; slow FT folded; steps unchanged',
                  note='No nn.bin copied: export from converted float32 state if needed')
    (output / 'conversion.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
