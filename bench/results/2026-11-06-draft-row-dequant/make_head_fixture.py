"""Make a named test fixture, not a model release: losslessly widen the real Q4 embedding/head codes to Q6."""
from pathlib import Path
import argparse
import hashlib
import json
import struct
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, required=True, help='original 4B model folder')
parser.add_argument('--out', type=Path, required=True, help='new fixture folder (must not exist)')
args = parser.parse_args()
src = args.source.resolve()
dst = args.out.resolve()
dst.mkdir(exist_ok=False)
name = 'language_model.model.embed_tokens.weight'
with (src / 'model.safetensors').open('rb') as f:
    old_hlen = struct.unpack('<Q', f.read(8))[0]
    old = json.loads(f.read(old_hlen))
new = json.loads(json.dumps(old))
N, W = old[name]['shape']
K = W * 8
new[name]['shape'] = [N, K * 6 // 32]
offset = 0
for key, tensor in new.items():
    if key == '__metadata__':
        continue
    size = N * K * 6 // 8 if key == name else old[key]['data_offsets'][1] - old[key]['data_offsets'][0]
    tensor['data_offsets'] = [offset, offset + size]
    offset += size
header = json.dumps(new, separators=(',', ':')).encode()
header += b' ' * (-len(header) % 8)
with (src / 'model.safetensors').open('rb') as inp, (dst / 'model.safetensors').open('wb') as out:
    out.write(struct.pack('<Q', len(header)))
    out.write(header)
    for key, tensor in new.items():
        if key == '__metadata__':
            continue
        begin, end = old[key]['data_offsets']
        inp.seek(8 + old_hlen + begin)
        if key != name:
            left = end - begin
            while left:
                chunk = inp.read(min(left, 8 << 20))
                assert chunk
                out.write(chunk)
                left -= len(chunk)
            continue
        for r0 in range(0, N, 256):
            rows = min(256, N - r0)
            q4 = np.frombuffer(inp.read(rows * W * 4), dtype='<u4').reshape(rows, W)
            codes = ((q4[..., None] >> (4 * np.arange(8, dtype=np.uint32))) & 15).reshape(rows, K)
            q6 = np.zeros((rows, K * 6 // 32), dtype='<u4')
            # Every 16 codes occupy exactly three words; split crossings explicitly.
            for j in range(16):
                shift = j * 6
                q6[:, shift // 32::3] |= codes[:, j::16] << (shift % 32)
                if shift % 32 + 6 > 32:
                    q6[:, shift // 32 + 1::3] |= codes[:, j::16] >> (32 - shift % 32)
            # Decode every widened code independently to verify the transformation before writing it.
            for k in range(K):
                bit = k * 6
                got = q6[:, bit // 32] >> (bit % 32)
                if bit % 32 + 6 > 32:
                    got = got | (q6[:, bit // 32 + 1] << (32 - bit % 32))
                assert np.array_equal(got & 63, codes[:, k])
            out.write(q6.tobytes())
config = json.loads((src / 'config.json').read_text())
for field in ('quantization', 'quantization_config'):
    for module in ('language_model.model.embed_tokens', 'model.embed_tokens', 'embed_tokens'):
        config[field][module] = {'bits': 6, 'group_size': 64, 'mode': 'affine'}
(dst / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
for path in src.iterdir():
    if path.name not in ('config.json', 'model.safetensors', 'model.safetensors.index.json'):
        (dst / path.name).symlink_to(path, target_is_directory=path.is_dir())
manifest = {'source_model': str(src), 'fixture': str(dst), 'changed_tensor': name, 'shape': [N, K],
            'source_bits': 4, 'fixture_bits': 6, 'group_size': 64, 'all_codes_equal': True,
            'unchanged': 'all scales, biases and other tensors; tokenizer and drafter are links to the source',
            'not_a_native_6bit_checkpoint': True,
            'config_sha256': hashlib.sha256((dst / 'config.json').read_bytes()).hexdigest()}
(dst / 'fixture.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))
