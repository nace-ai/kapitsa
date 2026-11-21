"""CPU mapping tests; optional local receipts read metadata, never sparse weights."""

import copy
import json
import os
from pathlib import Path
import unittest

import numpy as np

try:
    from .mapping import axis_permutation, map_tensor, skip_reason, transform_affine, transform_tensor
except ImportError:
    from mapping import axis_permutation, map_tensor, skip_reason, transform_affine, transform_tensor


CONFIG = {
    'model_type': 'qwen3_5',
    '_k708': {'linear_v_order': 'tiled'},
    'text_config': {
        'model_type': 'qwen3_5_text', 'num_hidden_layers': 4,
        'hidden_size': 64, 'vocab_size': 128, 'intermediate_size': 128,
        'num_attention_heads': 2, 'num_key_value_heads': 1, 'head_dim': 32,
        'full_attention_interval': 4, 'attn_output_gate': True,
        'linear_num_key_heads': 2, 'linear_num_value_heads': 6,
        'linear_key_head_dim': 32, 'linear_value_head_dim': 128,
        'linear_conv_kernel_dim': 4,
    },
}


def mapping(name, numpy_shape, config=CONFIG):
    return map_tensor(name, numpy_shape[::-1], config)


def tile(values, nk, ratio, width, axis=0):
    """Forward layout from llama.cpp conversion/qwen.py, independently expressed."""
    shape = values.shape
    split = shape[:axis] + (nk, ratio, width) + shape[axis + 1:]
    return values.reshape(split).swapaxes(axis, axis + 1).reshape(shape)


class MappingTests(unittest.TestCase):
    def test_global_names_and_no_transpose(self):
        for source, target in [('token_embd.weight', 'model.embed_tokens.weight'),
                               ('output.weight', 'lm_head.weight')]:
            m = mapping(source, (128, 64))
            self.assertEqual(m.mlx_name, 'language_model.' + target)
            self.assertEqual(m.source_shape, (128, 64))
            self.assertTrue(m.quantizable)
            values = np.arange(128 * 64).reshape(128, 64)
            np.testing.assert_array_equal(transform_tensor(values, m), values)

    def test_full_attention_q_gate_and_rope_are_unchanged(self):
        m = mapping('blk.3.attn_q.weight', (128, 64))
        values = np.arange(128 * 64).reshape(2, 2, 32, 64)
        # Per-head [query, gate] stays fused; this is not [all Q, all gates].
        np.testing.assert_array_equal(transform_tensor(values.reshape(128, 64), m), values.reshape(128, 64))
        self.assertEqual(m.transforms, ())
        self.assertEqual(m.mlx_name, 'language_model.model.layers.3.self_attn.q_proj.weight')
        m = mapping('blk.3.attn_k.weight', (32, 64))
        self.assertEqual(m.transforms, ())

    def test_norms_are_already_sanitized(self):
        for name, shape in [('output_norm.weight', (64,)), ('blk.0.attn_norm.weight', (64,)),
                            ('blk.0.post_attention_norm.weight', (64,)),
                            ('blk.3.attn_q_norm.weight', (32,)), ('blk.3.attn_k_norm.weight', (32,)),
                            ('blk.0.ssm_norm.weight', (128,))]:
            m = mapping(name, shape)
            self.assertFalse(m.quantizable)
            values = np.linspace(-2, 3, shape[0], dtype=np.float32)
            np.testing.assert_array_equal(transform_tensor(values, m), values)

    def test_linear_row_and_column_inverse(self):
        cases = [('attn_gate.weight', (768, 64), 0, 128),
                 ('ssm_alpha.weight', (6, 64), 0, 1),
                 ('ssm_beta.weight', (6, 64), 0, 1),
                 ('ssm_out.weight', (64, 768), 1, 128),
                 ('ssm_dt.bias', (6,), 0, 1)]
        for suffix, shape, axis, width in cases:
            with self.subTest(suffix=suffix):
                values = np.arange(np.prod(shape)).reshape(shape)
                gguf = tile(values, 2, 3, width, axis)
                m = mapping('blk.0.' + suffix, shape)
                np.testing.assert_array_equal(transform_tensor(gguf, m), values)
                self.assertIsNotNone(axis_permutation(m, axis))

    def test_qkv_and_conv_only_reorder_v(self):
        for suffix, shape in [('attn_qkv.weight', (896, 64)), ('ssm_conv1d.weight', (896, 4))]:
            with self.subTest(suffix=suffix):
                values = np.arange(np.prod(shape)).reshape(shape)
                prefix = 128  # Q + K = 2 * 2 key heads * 32 channels
                gguf = np.concatenate((values[:prefix], tile(values[prefix:], 2, 3, 128)))
                m = mapping('blk.0.' + suffix, shape)
                wanted = values[..., None] if 'conv' in suffix else values
                np.testing.assert_array_equal(transform_tensor(gguf, m), wanted)
                self.assertEqual(m.target_shape, wanted.shape)

    def test_a_log_inverse(self):
        values = np.linspace(-2, 2, 6, dtype=np.float32)
        gguf = tile(-np.exp(values), 2, 3, 1)
        m = mapping('blk.0.ssm_a', (6,))
        np.testing.assert_allclose(transform_tensor(gguf, m), values, atol=2e-7)
        for bad in (np.zeros(6), np.full(6, np.nan), np.full(6, np.inf)):
            with self.assertRaises(ValueError):
                transform_tensor(bad, m)

    def test_grouped_converter_is_explicit_identity(self):
        config = copy.deepcopy(CONFIG)
        config['_k708']['linear_v_order'] = 'grouped'
        m = mapping('blk.0.attn_gate.weight', (768, 64), config)
        self.assertEqual(m.transforms, ())
        del config['_k708']
        with self.assertRaisesRegex(ValueError, 'verify and set'):
            mapping('blk.0.attn_gate.weight', (768, 64), config)

    def test_skip_reasons(self):
        for name in ('blk.4.attn_q.weight', 'blk.4.nextn.eh_proj.weight', 'mtp.fc.weight'):
            self.assertIn('MTP', skip_reason(name, CONFIG))
            self.assertIsNone(map_tensor(name, (1,), CONFIG))
        for name in ('v.blk.0.attn_q.weight', 'vision_tower.patch_embed.weight', 'mm.0.weight'):
            self.assertIn('multimodal', skip_reason(name, CONFIG))
            self.assertIsNone(map_tensor(name, (1,), CONFIG))
        self.assertEqual(skip_reason('blk.0.surprise.weight', CONFIG), 'unsupported tensor name')

    def test_invalid_shapes_and_attention_kinds_fail(self):
        with self.assertRaisesRegex(ValueError, 'expected'):
            mapping('output.weight', (64, 128))
        with self.assertRaisesRegex(ValueError, 'attention kind'):
            mapping('blk.0.attn_q.weight', (128, 64))
        with self.assertRaisesRegex(ValueError, 'attention kind'):
            mapping('blk.3.attn_qkv.weight', (896, 64))
        m = mapping('output.weight', (128, 64))
        with self.assertRaisesRegex(ValueError, 'expected'):
            transform_tensor(np.zeros((64, 128)), m)


class AffineLayoutTests(unittest.TestCase):
    def test_packed_transforms_equal_unpacked_transforms(self):
        try:
            if __package__:
                from .quantization import pack, unpack
            else:
                from quantization import pack, unpack
        except ImportError as error:
            self.skipTest(f'quantization dependency unavailable: {error}')
        rng = np.random.default_rng(708)
        for suffix, shape in [('attn_qkv.weight', (896, 64)), ('attn_gate.weight', (768, 64)),
                              ('ssm_alpha.weight', (6, 64)), ('ssm_beta.weight', (6, 64)),
                              ('ssm_out.weight', (64, 768)), ('ffn_up.weight', (128, 64))]:
            m = mapping('blk.0.' + suffix, shape)
            for bits in (3, 4, 5, 6, 8):
                for group in (32, 64):
                    with self.subTest(suffix=suffix, bits=bits, group=group):
                        codes = rng.integers(0, 2**bits, size=shape, dtype=np.uint8)
                        scales = rng.uniform(0.1, 2, (shape[0], shape[1] // group)).astype(np.float16)
                        biases = rng.uniform(-1, 1, scales.shape).astype(np.float16)
                        packed = pack(codes, bits)
                        p, s, b = transform_affine(packed, scales, biases, bits, group, m)
                        np.testing.assert_array_equal(unpack(p, bits), transform_tensor(codes, m))
                        original = codes.astype(np.float32) * np.repeat(scales.astype(np.float32), group, axis=1)
                        original += np.repeat(biases.astype(np.float32), group, axis=1)
                        transformed = unpack(p, bits).astype(np.float32) * np.repeat(s.astype(np.float32), group, axis=1)
                        transformed += np.repeat(b.astype(np.float32), group, axis=1)
                        np.testing.assert_array_equal(transformed, transform_tensor(original, m))
                        if not m.transforms:
                            self.assertIs(p, packed)
                            self.assertIs(s, scales)
                            self.assertIs(b, biases)

    def test_affine_shape_guards(self):
        m = mapping('blk.0.ffn_up.weight', (128, 64))
        p = np.zeros((128, 8), np.uint32)
        s = np.zeros((128, 2), np.float16)
        with self.assertRaisesRegex(ValueError, 'packed'):
            transform_affine(p[:, :-1], s, s, 4, 32, m)
        with self.assertRaisesRegex(ValueError, 'coefficient'):
            transform_affine(p, s[:, :-1], s, 4, 32, m)
        with self.assertRaisesRegex(ValueError, 'group size'):
            transform_affine(p, s, s, 4, 128, m)
        with self.assertRaisesRegex(ValueError, 'requires'):
            transform_affine(p, s, s, 4, 32, mapping('blk.0.ssm_a', (6,)))
        for bits in (True, 4.0, 7):
            with self.assertRaisesRegex(ValueError, 'bit width'):
                transform_affine(p, s, s, bits, 32, m)
        with self.assertRaisesRegex(ValueError, 'group size'):
            transform_affine(p, s, s, 4, 32.0, m)

    def test_column_permutation_must_preserve_affine_groups(self):
        config = copy.deepcopy(CONFIG)
        config['text_config']['linear_value_head_dim'] = 32
        m = mapping('blk.0.ssm_out.weight', (64, 192), config)
        p = np.zeros((64, 24), np.uint32)
        s = np.zeros((64, 3), np.float16)
        with self.assertRaisesRegex(ValueError, 'whole affine groups'):
            transform_affine(p, s, s, 4, 64, m)


class LocalMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        state = Path.home() / '.agents/state'
        cls.tensor_path = Path(os.environ.get('K708_TENSOR_METADATA', state / 'k708/tensors.json'))
        inputs = Path(os.environ.get('K708_MLX_INPUTS', state / 'k581/mlx-rehearsal/inputs'))
        paths = [cls.tensor_path, inputs / 'config.json', inputs / 'model.safetensors.index.json']
        if not all(p.is_file() for p in paths):
            raise unittest.SkipTest('local K708 tensor metadata and MLX input index unavailable')
        cls.tensors = json.loads(paths[0].read_text())
        cls.config = json.loads(paths[1].read_text())
        cls.config['_k708'] = {'linear_v_order': 'tiled'}
        cls.index = json.loads(paths[2].read_text())['weight_map']

    def test_every_text_tensor_and_affine_triplet_has_exact_coverage(self):
        mapped, skipped, quantized = {}, [], set()
        for name, qtype, shape, _offset, _bytes in self.tensors:
            m = map_tensor(name, tuple(shape), self.config)
            if m is None:
                skipped.append((name, skip_reason(name, self.config)))
                continue
            self.assertNotIn(m.mlx_name, mapped)
            mapped[m.mlx_name] = m
            self.assertIn(m.mlx_name, self.index)
            if qtype != 'F32':
                self.assertTrue(m.quantizable)
                quantized.add(m.module)
                for suffix in ('.weight', '.scales', '.biases'):
                    self.assertIn(m.module + suffix, self.index)
        text_keys = {name for name in self.index if name.startswith('language_model.')}
        wanted = set(mapped)
        for module in quantized:
            wanted.update((module + '.scales', module + '.biases'))
        self.assertSetEqual(wanted, text_keys)
        self.assertEqual(len(self.tensors), 866)
        self.assertEqual(len(mapped), 851)
        self.assertEqual(len(quantized), 498)
        self.assertEqual(len(skipped), 15)
        self.assertTrue(all(reason.startswith('MTP:') for _, reason in skipped))
        self.assertEqual(sum(name.startswith('vision_tower.') for name in self.index), 333)
        print(f'mapping coverage: {len(mapped)} text tensors, {len(quantized)} affine modules, '
              f'{len(skipped)} MTP tensors skipped; all {len(text_keys)} MLX text keys covered')

    def test_names_match_authoritative_gguf_tensor_map(self):
        try:
            import gguf
        except ImportError:
            self.skipTest('gguf package unavailable')
        names = gguf.get_tensor_name_map(gguf.MODEL_ARCH.QWEN35, 65)
        for name, _qtype, shape, _offset, _bytes in self.tensors:
            m = map_tensor(name, tuple(shape), self.config)
            if m is None:
                continue
            hf_name = m.mlx_name.removeprefix('language_model.')
            hf_name = hf_name.replace('.linear_attn.dt_bias', '.linear_attn.dt_proj.bias')
            self.assertEqual(names.get_name(hf_name, try_suffixes=('.weight', '.bias')), name)


if __name__ == '__main__':
    unittest.main(verbosity=2)
