"""Rounding with error feedback (block LDLQ, as in QuIP#) for the lattice codes, as a kill test.

The current fit rounds each vector of eight weights to its nearest lattice point, weighting inputs
by a diagonal. With the inputs' full second-moment matrix ``H = E[x x^T]`` the output error of a row
is ``e H e^T`` for ``e = w - w_hat``. Factor ``H = L D L^T`` with ``L`` block unit lower triangular
(blocks of eight) and round the blocks from last to first, each after adding the error already
made on later blocks, carried through ``L``: the loss becomes the sum of the blocks' own rounding
errors under ``D``, instead of those errors compounding through the inputs' correlations.

    python ldlq.py            # synthetic check: nearest rounding against feedback at three spectra
"""

from __future__ import annotations

import torch

from knee_codebooks import nearest


def plain(w: torch.Tensor, book: torch.Tensor, group: int, rounds: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearest-point rounding with one least-squares scale per (row, group): (w_hat, scale [out, groups])."""
    v = w.reshape(w.shape[0], -1, group // 8, 8)
    scale = v.pow(2).mean((-1, -2), keepdim=True).sqrt() / book.pow(2).sum(1).mean().div(8).sqrt()
    for _ in range(rounds):
        c = book[nearest((v / scale).reshape(-1, 8), book)].reshape(v.shape)
        scale = (v * c).sum((-1, -2), keepdim=True) / c.pow(2).sum((-1, -2), keepdim=True).clamp_min(1e-12)
    return (scale * c).reshape(w.shape), scale[:, :, 0, 0]


def ldlq(w: torch.Tensor, hessian: torch.Tensor, book: torch.Tensor, group: int, damp: float = 0.01) -> torch.Tensor:
    """``w`` ([out, in]) rounded block by block with feedback through ``hessian`` ([in, in]); scales from ``plain``."""
    n = w.shape[1]
    _, scale = plain(w, book, group)
    h = hessian.double() + damp * hessian.diagonal().mean().double() * torch.eye(n, dtype=torch.float64, device=w.device)
    chol = torch.linalg.cholesky(h)
    diag = torch.stack([chol[i:i + 8, i:i + 8] for i in range(0, n, 8)])  # [n/8, 8, 8]
    lower = torch.einsum("nbi,bij->nbj", chol.reshape(n, n // 8, 8), torch.linalg.inv(diag))
    lower = lower.reshape(n, n).to(w.dtype)  # identity blocks on the diagonal
    w_hat, err = torch.zeros_like(w), torch.zeros_like(w)
    for k in reversed(range(n // 8)):
        cols = slice(8 * k, 8 * k + 8)
        target = w[:, cols] + err[:, 8 * k + 8:] @ lower[8 * k + 8:, cols]
        s = scale[:, 8 * k // group, None]
        w_hat[:, cols] = book[nearest(target / s, book)] * s
        err[:, cols] = w[:, cols] - w_hat[:, cols]
    return w_hat


def output_error(w: torch.Tensor, w_hat: torch.Tensor, hessian: torch.Tensor) -> float:
    """``tr(E H E^T) / tr(W H W^T)``: the relative squared error of the projection's outputs."""
    e, h = (w - w_hat).double(), hessian.double()
    return float(((e @ h) * e).sum() / ((w.double() @ h) * w.double()).sum())


def demo() -> None:
    import numpy as np
    from keep.vq.e8 import e8p_full_grid

    torch.manual_seed(0)
    book = torch.from_numpy(e8p_full_grid().astype(np.float32))
    n, out = 512, 96
    w = torch.randn(out, n)
    eye = torch.eye(n)
    flat = output_error(w, plain(w, book, 256)[0], eye), output_error(w, ldlq(w, eye, book, 256), eye)
    assert abs(flat[0] - flat[1]) < 2e-3, f"with uncorrelated inputs feedback should change nothing: {flat}"
    print("input spectrum | nearest | feedback")
    for name, decay in (("flat", 0.0), ("power law 0.5", 0.5), ("power law 1.0", 1.0)):
        basis = torch.linalg.qr(torch.randn(n, n))[0]
        h = (basis * torch.arange(1, n + 1).float().pow(-decay)) @ basis.T
        a, b = output_error(w, plain(w, book, 256)[0], h), output_error(w, ldlq(w, h, book, 256), h)
        print(f"{name:>14} | {a:.4f} | {b:.4f}")
        assert b <= a * 1.001, (name, a, b)


if __name__ == "__main__":
    demo()
