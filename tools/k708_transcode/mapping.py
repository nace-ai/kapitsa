"""Qwen3.5 GGUF -> sanitized MLX skeleton names and CPU layout transforms.

Shapes passed to map_tensor are GGUF dimensions, innermost first. Arrays passed
into transform_tensor are already dequantized in NumPy order, the reverse of
those dimensions. Reversing the shape is NOT a matrix transpose.

The explicit name table mirrors gguf.TensorNameMap(QWEN35). Keeping the small
subset here avoids making gguf or torch a runtime dependency. Qwen3.5's llama.cpp
converter preserves full-attention Q/K rows and the per-head fused Q/gate layout;
it does not apply Llama's RoPE permutation. Its optional linear V-head reorder
must be verified against real weights, then selected with
config['_k708']['linear_v_order'] = 'grouped' or 'tiled'. No default guesses the
converter revision. The tiled inverse follows conversion/qwen.py's
_LinearAttentionVReorderBase.

Both llama.cpp and MLX sanitize add one to HF RMSNorm parameters. Consequently
GGUF -> shipped, sanitized MLX norms need no offset. Linear-attention norm is
excluded by both. MTP and multimodal tensors stay outside the text replacement.
"""

from dataclasses import dataclass
import re

import numpy as np


_GLOBAL_NAMES = {
    'token_embd.weight': 'language_model.model.embed_tokens.weight',
    'output.weight': 'language_model.lm_head.weight',
    'output_norm.weight': 'language_model.model.norm.weight',
}
_LAYER_NAMES = {
    'attn_norm.weight': 'input_layernorm.weight',
    'post_attention_norm.weight': 'post_attention_layernorm.weight',
    'ffn_gate.weight': 'mlp.gate_proj.weight',
    'ffn_up.weight': 'mlp.up_proj.weight',
    'ffn_down.weight': 'mlp.down_proj.weight',
    'attn_q.weight': 'self_attn.q_proj.weight',
    'attn_k.weight': 'self_attn.k_proj.weight',
    'attn_v.weight': 'self_attn.v_proj.weight',
    'attn_output.weight': 'self_attn.o_proj.weight',
    'attn_q_norm.weight': 'self_attn.q_norm.weight',
    'attn_k_norm.weight': 'self_attn.k_norm.weight',
    'attn_qkv.weight': 'linear_attn.in_proj_qkv.weight',
    'attn_gate.weight': 'linear_attn.in_proj_z.weight',
    'ssm_alpha.weight': 'linear_attn.in_proj_a.weight',
    'ssm_beta.weight': 'linear_attn.in_proj_b.weight',
    'ssm_out.weight': 'linear_attn.out_proj.weight',
    'ssm_a': 'linear_attn.A_log',
    'ssm_dt.bias': 'linear_attn.dt_bias',
    'ssm_conv1d.weight': 'linear_attn.conv1d.weight',
    'ssm_norm.weight': 'linear_attn.norm.weight',
}
_BLOCK = re.compile(r'blk\.(\d+)\.(.+)\Z')


@dataclass(frozen=True)
class Mapping:
    gguf_name: str
    mlx_name: str
    gguf_shape: tuple[int, ...]
    target_shape: tuple[int, ...]
    transforms: tuple[str, ...] = ()
    # key heads, value heads per key head, key head width, value head width
    linear_dims: tuple[int, int, int, int] | None = None
    quantizable: bool = False

    @property
    def source_shape(self) -> tuple[int, ...]:
        return self.gguf_shape[::-1]

    @property
    def module(self) -> str:
        return self.mlx_name.removesuffix('.weight')


def _text_config(config: dict) -> dict:
    text = config.get('text_config', config)
    if text.get('model_type') not in ('qwen3_5', 'qwen3_5_text'):
        raise ValueError('mapping supports dense Qwen3.5 text checkpoints only')
    return text


def skip_reason(gguf_name: str, config: dict) -> str | None:
    """Explain exclusions. Unknown text names are never silently accepted."""
    if gguf_name.startswith(('v.', 'mm.', 'a.', 'vision_', 'vision.', 'visual.',
                             'model.visual.', 'model.vision.', 'audio.',
                             'vision_tower.', 'multi_modal_projector.')):
        return 'multimodal: preserve shipped skeleton'
    text = _text_config(config)
    block = _BLOCK.fullmatch(gguf_name)
    if '.nextn.' in gguf_name or gguf_name.startswith(('mtp.', 'nextn.')):
        return 'MTP: absent from inference skeleton'
    if block and int(block[1]) >= int(text['num_hidden_layers']):
        return 'MTP: layer beyond text num_hidden_layers'
    if gguf_name in _GLOBAL_NAMES or block and block[2] in _LAYER_NAMES:
        return None
    return 'unsupported tensor name'


def map_tensor(gguf_name: str, shape: tuple, config: dict) -> Mapping | None:
    """Map one text tensor; reject bad shapes or unverified linear layouts.

    quantizable marks affine embedding/linear modules, not the GGUF storage type.
    Replace a mapping only when its skeleton module has .scales and .biases.
    Keep every other skeleton tensor unchanged, including mapped float tensors.
    """
    if skip_reason(gguf_name, config) is not None:
        return None
    text = _text_config(config)
    shape = tuple(int(n) for n in shape)
    hidden, vocab = int(text['hidden_size']), int(text['vocab_size'])
    transforms = []
    dims = None
    block = _BLOCK.fullmatch(gguf_name)
    if block is None:
        name = _GLOBAL_NAMES[gguf_name]
        expected = (hidden,) if gguf_name == 'output_norm.weight' else (vocab, hidden)
    else:
        layer, suffix = int(block[1]), block[2]
        target = _LAYER_NAMES[suffix]
        name = f'language_model.model.layers.{layer}.{target}'
        layer_types = text.get('layer_types')
        linear = (layer_types[layer] == 'linear_attention' if layer_types else
                  (layer + 1) % int(text['full_attention_interval']) != 0)
        if target.startswith('linear_attn.') != linear and target.startswith(('linear_attn.', 'self_attn.')):
            raise ValueError(f'{gguf_name}: attention kind disagrees with config')
        intermediate = int(text['intermediate_size'])
        head_dim = int(text.get('head_dim') or hidden // int(text['num_attention_heads']))
        qdim = int(text['num_attention_heads']) * head_dim
        kvdim = int(text['num_key_value_heads']) * head_dim
        if target.startswith('linear_attn.'):
            nk = int(text['linear_num_key_heads'])
            nv = int(text['linear_num_value_heads'])
            dk, dv = int(text['linear_key_head_dim']), int(text['linear_value_head_dim'])
            if nk <= 0 or nv <= 0 or nv % nk:
                raise ValueError('linear value heads must be divisible by key heads')
            dims = (nk, nv // nk, dk, dv)
            key, value = nk * dk, nv * dv
            conv = 2 * key + value
            shapes = {
                'attn_qkv.weight': (conv, hidden), 'attn_gate.weight': (value, hidden),
                'ssm_alpha.weight': (nv, hidden), 'ssm_beta.weight': (nv, hidden),
                'ssm_out.weight': (hidden, value), 'ssm_a': (nv,), 'ssm_dt.bias': (nv,),
                'ssm_conv1d.weight': (conv, int(text['linear_conv_kernel_dim'])),
                'ssm_norm.weight': (dv,),
            }
            expected = shapes[suffix]
            if suffix != 'ssm_norm.weight' and nv != nk:
                order = config.get('_k708', {}).get('linear_v_order')
                if order not in ('grouped', 'tiled'):
                    raise ValueError('verify and set config["_k708"]["linear_v_order"] to grouped or tiled')
                if order == 'tiled':
                    if suffix in ('attn_qkv.weight', 'ssm_conv1d.weight'):
                        transforms.append('untile_qkv_rows')
                    elif suffix == 'ssm_out.weight':
                        transforms.append('untile_v_columns')
                    else:
                        transforms.append('untile_v_rows')
            if suffix == 'ssm_a':
                transforms.append('log_negative')
            if suffix == 'ssm_conv1d.weight':
                transforms.append('conv_mlx')
        else:
            shapes = {
                'attn_norm.weight': (hidden,), 'post_attention_norm.weight': (hidden,),
                'ffn_gate.weight': (intermediate, hidden), 'ffn_up.weight': (intermediate, hidden),
                'ffn_down.weight': (hidden, intermediate),
                'attn_q.weight': (qdim * (2 if text.get('attn_output_gate', True) else 1), hidden),
                'attn_k.weight': (kvdim, hidden), 'attn_v.weight': (kvdim, hidden),
                'attn_output.weight': (hidden, qdim),
                'attn_q_norm.weight': (head_dim,), 'attn_k_norm.weight': (head_dim,),
            }
            expected = shapes[suffix]
    if shape[::-1] != expected:
        raise ValueError(f'{gguf_name}: NumPy shape {shape[::-1]}, expected {expected}')
    target_shape = expected + (1,) if 'conv_mlx' in transforms else expected
    return Mapping(gguf_name, name, shape, target_shape, tuple(transforms), dims,
                   len(expected) == 2 and 'conv_mlx' not in transforms)


def axis_permutation(mapping: Mapping, axis: int) -> np.ndarray | None:
    """Return inverse V-head row/column indices, or None for identity.

    Row permutations also apply unchanged to packed codes, scales and biases.
    Column permutations act on unpacked codes. For group-aligned head widths,
    scales/biases use permutation.reshape(-1, group_size)[:, 0] // group_size.
    """
    if axis not in (0, 1):
        raise ValueError('only row or column permutations are supported')
    transforms = mapping.transforms
    qkv = axis == 0 and 'untile_qkv_rows' in transforms
    columns = axis == 1 and 'untile_v_columns' in transforms
    rows = axis == 0 and 'untile_v_rows' in transforms
    if not (qkv or columns or rows):
        return None
    nk, ratio, dk, dv = mapping.linear_dims
    width = 1 if mapping.gguf_name.endswith(('ssm_a', 'ssm_dt.bias', 'ssm_alpha.weight', 'ssm_beta.weight')) else dv
    perm = np.arange(nk * ratio * width).reshape(ratio, nk, width).swapaxes(0, 1).reshape(-1)
    if qkv:
        prefix = 2 * nk * dk
        perm = np.concatenate((np.arange(prefix), prefix + perm))
    return perm


def transform_affine(packed: np.ndarray, scales: np.ndarray, biases: np.ndarray,
                     bits: int, group_size: int, mapping: Mapping) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the verified layout inverse without unpacking or changing codes."""
    if not mapping.quantizable or len(mapping.source_shape) != 2:
        raise ValueError('affine transform requires an embedding or linear weight')
    rows, columns = mapping.source_shape
    if not isinstance(bits, (int, np.integer)) or isinstance(bits, (bool, np.bool_)) or bits not in (2, 3, 4, 5, 6, 8):
        raise ValueError('unsupported affine bit width')
    if not isinstance(group_size, (int, np.integer)) or group_size not in (32, 64) or columns % group_size or columns % 32:
        raise ValueError('affine columns must be divisible by group size 32 or 64')
    packed, scales, biases = np.asarray(packed), np.asarray(scales), np.asarray(biases)
    if packed.dtype != np.uint32 or packed.shape != (rows, columns // 32 * bits):
        raise ValueError('packed uint32 shape disagrees with mapping and bit width')
    if scales.shape != (rows, columns // group_size) or biases.shape != scales.shape:
        raise ValueError('affine coefficient shape disagrees with mapping and group size')
    row_perm = axis_permutation(mapping, 0)
    col_perm = axis_permutation(mapping, 1)
    if row_perm is not None:
        packed, scales, biases = (np.take(a, row_perm, axis=0) for a in (packed, scales, biases))
    if col_perm is not None:
        # A 32-code block occupies exactly bits uint32 words, including 3/5/6 bits.
        groups = col_perm.reshape(-1, group_size)
        starts = groups[:, 0]
        if np.any(starts % group_size) or not np.array_equal(groups, starts[:, None] + np.arange(group_size)):
            raise ValueError('column permutation does not preserve whole affine groups')
        blocks = col_perm.reshape(-1, 32)[:, 0] // 32
        word_perm = (blocks[:, None] * bits + np.arange(bits)).reshape(-1)
        packed = np.take(packed, word_perm, axis=1)
        scales, biases = (np.take(a, starts // group_size, axis=1) for a in (scales, biases))
    return packed, scales, biases


def transform_tensor(array: np.ndarray, mapping: Mapping) -> np.ndarray:
    """Transform actual dequantized values into sanitized MLX tensor order."""
    result = np.asarray(array)
    if result.shape != mapping.source_shape:
        raise ValueError(f'{mapping.gguf_name}: got {result.shape}, expected {mapping.source_shape}')
    for axis in (0, 1):
        perm = axis_permutation(mapping, axis)
        if perm is not None:
            result = np.take(result, perm, axis=axis)
    if 'log_negative' in mapping.transforms:
        if np.any(result >= 0) or not np.all(np.isfinite(result)):
            raise ValueError('GGUF ssm_a must be finite and strictly negative')
        result = np.log(-result.astype(np.float32))
    if 'conv_mlx' in mapping.transforms:
        result = result[..., None]
    return np.ascontiguousarray(result)
