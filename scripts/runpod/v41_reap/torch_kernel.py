"""Plain-torch stand-ins for the reference ``inference/kernel.py`` (tilelang, GPU only).

Install as ``sys.modules["kernel"]`` before importing DeepSeek's ``model.py``. Two uses:

- CPU tests at toy size, with bf16 weights. The in-place quantizers stay no-ops there, because toy
  head sizes are not multiples of the quantization blocks.
- The real FP8/FP4 checkpoint on a machine without the GPU kernels (the M4 Max). Set
  ``exact_quant = True``. Every function is then a port of the kernel's math: the same block
  scales, power-of-2 rounding and clamps. The GEMMs dequantize their operands and multiply in
  fp32, so only the summation order differs from the kernels.

``sparse_attn`` and ``hc_split_sinkhorn`` are always exact ports.
"""

from __future__ import annotations

import torch


exact_quant = False  # True: in-place quantizers fake-quantize like the GPU kernels (needed for real weights)

FP8_MAX = 448.0
FP4_MAX = 6.0
# e2m1 magnitudes by 3-bit code; bit 3 of the nibble is the sign
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _pow2_ceil(v: torch.Tensor) -> torch.Tensor:
    """2 ** ceil(log2(v)) for positive fp32 v, by the kernel's bit trick (fast_round_scale)."""
    bits = v.float().contiguous().view(torch.int32)
    exp = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    return ((exp + 127) << 23).view(torch.float32)


def _blocks(x: torch.Tensor, block_size: int) -> torch.Tensor:
    assert x.size(-1) % block_size == 0, (x.shape, block_size)
    return x.float().unflatten(-1, (-1, block_size))  # [..., n_blocks, block]


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    """Block-wise FP8 e4m3 quantization of activations (``act_quant_kernel``)."""
    if inplace and not exact_quant:
        return x
    xb = _blocks(x, block_size)
    amax = xb.abs().amax(dim=-1).clamp_min(1e-4)
    s = amax * torch.tensor(1 / FP8_MAX, dtype=torch.float32)
    if scale_fmt is not None:
        s = _pow2_ceil(s)
    y = (xb / s[..., None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    if inplace:
        x.copy_((y.float() * s[..., None]).flatten(-2).to(x.dtype))
        return x
    return y.flatten(-2), s.to(scale_dtype)


def _e2m1_round(v: torch.Tensor) -> torch.Tensor:
    """Nearest e2m1 value, ties to the even code (``T.Cast(FP4, ...)``); v already in [-6, 6]."""
    mag = v.abs()
    grid = E2M1.to(v.device)
    dist = (mag[..., None] - grid).abs()
    best = dist.amin(dim=-1, keepdim=True)
    codes = torch.arange(8, device=v.device)
    # among equally near codes take the even one (its mantissa bit is 0)
    pick = torch.where(dist == best, codes % 2, 99).argmin(dim=-1)  # ties are adjacent codes: one even
    return torch.copysign(grid[pick], v)


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    """Block-wise FP4 fake quantization (``fp4_quant_kernel``); only the in-place form is used."""
    assert inplace, "model.py only calls fp4_act_quant in place"
    if not exact_quant:
        return x
    xb = _blocks(x, block_size)
    amax = xb.abs().amax(dim=-1)
    if scale_dtype == torch.float8_e4m3fn:
        s = (amax.clamp_min(6 * 2**-9) / FP4_MAX).to(torch.float8_e4m3fn).float()
    else:
        s = _pow2_ceil(amax.clamp_min(6 * 2**-126) * torch.tensor(1 / FP4_MAX, dtype=torch.float32))
    y = _e2m1_round((xb / s[..., None]).clamp(-FP4_MAX, FP4_MAX))
    x.copy_((y * s[..., None]).flatten(-2).to(x.dtype))
    return x


def _dequant_rows(values: torch.Tensor, scale: torch.Tensor, block: int) -> torch.Tensor:
    """values [R, K] fp32 times a per-(row, K-block) scale [R, K/block]."""
    return (values.unflatten(-1, (-1, block)) * scale.float()[..., None]).flatten(-2)


def decode_e2m1(b: torch.Tensor) -> torch.Tensor:
    """float4_e2m1fn_x2 [N, K/2] to fp32 [N, K]; the low nibble is the first element."""
    codes = b.contiguous().view(torch.uint8)
    nib = torch.stack((codes & 0x0F, codes >> 4), dim=-1).flatten(-2).long()
    mag = E2M1.to(b.device)[nib & 0x7]
    return torch.where((nib & 0x8) != 0, -mag, mag)


def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, block_size=128):
    """C[M,N] = A[M,K] @ B[N,K]^T; A scaled per (row, K block), B per (N block, K block)."""
    K = a.size(-1)
    A = _dequant_rows(a.reshape(-1, K).float(), a_s.reshape(-1, K // block_size), block_size)
    sb = b_s.float().repeat_interleave(block_size, dim=0)[: b.size(0)]
    B = _dequant_rows(b.float(), sb, block_size)
    return (A @ B.T).to(torch.get_default_dtype()).view(*a.shape[:-1], b.size(0))


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, act_block_size=128):
    """C[M,N] = A_fp8[M,K] @ B_fp4[N,K]^T; B scaled per 32 along K (e8m0)."""
    K = a.size(-1)
    A = _dequant_rows(a.reshape(-1, K).float(), a_s.reshape(-1, K // act_block_size), act_block_size)
    B = _dequant_rows(decode_e2m1(b), b_s, 32)
    return (A @ B.T).to(torch.get_default_dtype()).view(*a.shape[:-1], b.size(0))


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """q [b,m,h,d], kv [b,n,d] (one shared head), topk_idxs [b,m,k] with -1 = empty slot.
    Softmax over the gathered slots plus a per-head sink logit; a row with no valid slot is 0."""

    b = q.shape[0]
    valid = topk_idxs >= 0
    gathered = kv.float()[torch.arange(b)[:, None, None], topk_idxs.clamp_min(0).long()]  # [b,m,k,d]
    scores = torch.einsum("bmhd,bmkd->bmhk", q.float(), gathered) * softmax_scale
    scores = scores.masked_fill(~valid[:, :, None, :], -torch.inf)
    row_max = scores.amax(dim=-1, keepdim=True).clamp_min(-1e30)  # the kernel's finite floor
    p = torch.exp(scores - row_max)
    denom = p.sum(dim=-1) + torch.exp(attn_sink.float()[None, None, :] - row_max[..., 0])
    o = torch.einsum("bmhk,bmkd->bmhd", p, gathered) / denom[..., None]
    return o.to(q.dtype)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    hc = hc_mult
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
    comb = (mixes[..., 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]).unflatten(-1, (hc, hc))
    comb = comb.softmax(dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb
