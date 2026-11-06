"""Plain-torch stand-ins for the reference ``inference/kernel.py`` (tilelang, GPU only).

Used by the CPU tests to run DeepSeek's real ``model.py`` at toy size: install it
as ``sys.modules["kernel"]`` before importing ``model``. It covers the bf16 path
only. The quantizers are no-ops (they fake-quantize in place on the GPU) and the
quantized GEMMs refuse to run, so a test model must use ``dtype="bf16"`` and
``expert_dtype=None``. ``sparse_attn`` and ``hc_split_sinkhorn`` are exact ports
of the kernels' math.
"""

from __future__ import annotations

import torch


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    if inplace:
        return x
    raise NotImplementedError("fp8 activations: CPU tests use bf16 weights")


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=None):
    if inplace:
        return x
    raise NotImplementedError("fp4 activations: CPU tests use bf16 weights")


def fp8_gemm(*args, **kwargs):
    raise NotImplementedError("fp8 GEMM: CPU tests use bf16 weights")


def fp4_gemm(*args, **kwargs):
    raise NotImplementedError("fp4 GEMM: CPU tests use bf16 weights")


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
