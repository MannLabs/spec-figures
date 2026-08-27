"""Pull the embedded raster images out of Figure1.ai — PROVISIONAL input for c."""
import re
import sys
import zlib
from pathlib import Path
import numpy as np
from PIL import Image
src = Path(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
raw = src.read_bytes()
print(f'{src.name}: {len(raw)/1e6:.1f} MB')
obj_re = re.compile(rb'(\d+)\s+(\d+)\s+obj\b')
def get(d, key, default=None):
    """PDF writers omit the space before a name value: /Filter/FlateDecode."""
    m = re.search(rb'/' + key + rb'\s*\[?\s*(/?[\w.\-]+)', d)
    return m.group(1) if m else default
rows = []
for m in obj_re.finditer(raw):
    start = m.end()
    s = raw.find(b'stream', start)
    if s < 0 or s - start > 4000:
        continue
    header = raw[start:s]
    if b'/Image' not in header:
        continue
    data_start = s + 6
    while raw[data_start:data_start + 1] in (b'\r', b'\n'):
        data_start += 1
    e = raw.find(b'endstream', data_start)
    payload = raw[data_start:e]
    num = int(m.group(1))
    w = get(header, b'Width'); h = get(header, b'Height')
    bpc = get(header, b'BitsPerComponent'); flt = get(header, b'Filter')
    cs = get(header, b'ColorSpace')
    if w is None or h is None:
        continue
    w, h = int(w), int(h)
    bpc = int(bpc) if bpc else 8
    rows.append((num, w, h, bpc, flt, cs, len(payload)))
    stem = out / f'obj{num:05d}_{w}x{h}_{bpc}bpc'
    try:
        if flt == b'/DCTDecode':
            stem.with_suffix('.jpg').write_bytes(payload)
        elif flt == b'/FlateDecode':
            dec = zlib.decompress(payload)
            ncomp = max(1, round(len(dec) * 8 / (w * h * bpc)))
            if bpc == 8 and ncomp in (1, 3, 4):
                arr = np.frombuffer(dec[:w * h * ncomp], np.uint8).reshape(h, w, ncomp)
                if ncomp == 4:
                    np.save(stem.with_suffix('.npy'), arr)
                    Image.fromarray(arr, 'CMYK').convert('RGB').save(
                        stem.with_suffix('.png'))
                else:
                    Image.fromarray(arr.squeeze(), 'RGB' if ncomp == 3 else 'L').save(
                        stem.with_suffix('.png'))
            elif bpc == 16 and ncomp == 1:
                arr = np.frombuffer(dec[:w * h * 2], '>u2').reshape(h, w)
                np.save(stem.with_suffix('.npy'), arr)
            else:
                stem.with_suffix('.bin').write_bytes(dec)
        else:
            stem.with_suffix('.bin').write_bytes(payload)
    except Exception as exc:
        print('  failed', num, exc)
print(f'{len(rows)} image objects')
for r in sorted(rows, key=lambda r: -r[1] * r[2]):
    print(f'  obj {r[0]:6d}  {r[1]:5d} x {r[2]:5d}  {r[3]:2d} bpc  '
          f'{(r[4] or b"-").decode():14s} {(r[5] or b"-").decode():12s} '
          f'{r[6]/1e6:7.2f} MB')
