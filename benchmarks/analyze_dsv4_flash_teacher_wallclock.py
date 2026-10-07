"""Extrapolate measured DeepSeek-V4-Flash slice timings to a full teacher run.

Consumes the JSON written by
``benchmarks/bench_dsv4_flash_layer_slice_prefill.py`` and reports, for each
batch size measured, the wall-clock of forwarding the whole teich corpus through
all 43 backbone layers -- integrated over the **real** session-length
distribution in the split manifest rather than a single average length, because
the ratio-4 indexer term is quadratic in sequence length.

The real schedule has 43 ``compress_ratios`` entries: ratio 0 at layers 0, 1 and
42 (3 local), ratio 4 at layers 2, 4, ..., 40 (20 sparse), ratio 128 at layers
3, 5, ..., 41 (20 compressed). All ratio-4 layers pool identically (L/4) and all
ratio-128 layers pool identically (L/128), so one measured sample of each is
representative::

    T_43(L) = 3*t_local + 20*t_sparse4 + 20*t_comp128 + t_head + t_embed

One approximation is folded in and is small: the measured ratio-4 layer (slice
layer 2) is hash-routed while 19 of the 20 real ones are scored. Both branches
run the same ``x @ gate.weight.T``; the difference is an ``argpartition`` over
256 against an int64 gather, well under 1% of a 15 s layer.

Two estimators are reported and should agree: piecewise-linear interpolation of
per-token cost through the measured lengths, and a least-squares fit
``c(L) = a + b*L`` (every term is linear in tokens except the indexer, which
scores every pooled window for every query and is therefore linear in position).

Usage::

    uv run --group dev python benchmarks/analyze_dsv4_flash_teacher_wallclock.py \\
        /tmp/slice-*.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "recipes/dsv4_teich_split_manifest_v1_20260811.json"

N_LOCAL, N_SPARSE4, N_COMP128 = 3, 20, 20
#: measured from the 48 shard headers at revision 7872f01b
ROUTED_BYTES_BACKBONE = 147_169_738_752
#: measured: 25 GB across 7 contiguous shards, cold, F_NOCACHE
DEFAULT_SSD_GB_PER_S = 8.129


def full_model_seconds(record: dict) -> dict:
    layer = record["layer_seconds"]
    local = (layer[0] + layer[1]) / 2
    barriered = sum(layer) + record["head_seconds"] + record["embed_seconds"]
    # Anchor the extrapolation on the wall-clock the unbarriered pass actually
    # took, so any residual barrier cost is divided back out.
    scale = record["total_seconds_unbarriered"] / barriered
    total = (
        N_LOCAL * local
        + N_SPARSE4 * layer[2]
        + N_COMP128 * layer[3]
        + record["head_seconds"]
        + record["embed_seconds"]
    ) * scale
    return {
        "length": record["length"],
        "batch": record["batch"],
        "chunk": record["chunk"],
        "tokens": record["tokens"],
        "slice_seconds": record["total_seconds_unbarriered"],
        "slice_tok_s": record["tokens"] / record["total_seconds_unbarriered"],
        "t_local": local,
        "t_sparse4": layer[2],
        "t_comp128": layer[3],
        "full43_seconds": total,
        "full43_tok_s": record["tokens"] / total,
        "peak_gb": record.get("peak_gb"),
    }


def interpolate(curve: list[dict], length: int) -> float:
    points = sorted((c["length"], c["full43_seconds"] / c["tokens"]) for c in curve)
    if length <= points[0][0]:
        return points[0][1] * length
    if length >= points[-1][0]:
        return points[-1][1] * length
    for (l0, c0), (l1, c1) in zip(points, points[1:]):
        if l0 <= length <= l1:
            return (c0 + (length - l0) / (l1 - l0) * (c1 - c0)) * length
    raise AssertionError("unreachable")


def fit_linear_per_token(curve: list[dict]) -> tuple[float, float, float]:
    xs = [c["length"] for c in curve]
    ys = [c["full43_seconds"] / c["tokens"] for c in curve]
    n = len(xs)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    denom = sum((x - mean_x) ** 2 for x in xs)
    b = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom
    a = mean_y - b * mean_x
    worst = max(abs(y - (a + b * x)) / y for x, y in zip(xs, ys))
    return a, b, worst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", type=Path, nargs="+")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--ssd-gb-per-s", type=float, default=DEFAULT_SSD_GB_PER_S)
    opts = ap.parse_args()

    grouped: dict[tuple, list[dict]] = {}
    for path in opts.results:
        for record in json.loads(path.read_text())["results"]:
            key = (record["batch"], record["chunk"], record["length"])
            grouped.setdefault(key, []).append(record)

    curves: dict[tuple[int, int], list[dict]] = {}
    for (batch, chunk, _length), records in sorted(grouped.items()):
        # Several samples of one configuration: keep the fastest. Run-to-run
        # spread on this machine is ~12% and the slow tail is contention, not
        # the model.
        best = min(records, key=lambda r: r["total_seconds_unbarriered"])
        curves.setdefault((batch, chunk), []).append(full_model_seconds(best))

    manifest = json.loads(opts.manifest.read_text())
    splits: dict[str, list[int]] = {}
    for row in manifest["rows"]:
        splits.setdefault(row["campaign_split"], []).append(row["v4_token_count"])
    all_lengths = [n for v in splits.values() for n in v]

    header = f"{'B':>2} {'chunk':>5} {'L':>7} {'slice t/s':>9} {'T43 s':>8} {'T43 t/s':>8} {'GB':>5}"
    print(header)
    for (batch, chunk), curve in sorted(curves.items()):
        for c in sorted(curve, key=lambda c: c["length"]):
            print(f"{batch:>2} {chunk:>5} {c['length']:>7} {c['slice_tok_s']:>9.0f} "
                  f"{c['full43_seconds']:>8.1f} {c['full43_tok_s']:>8.1f} "
                  f"{c['peak_gb'] or 0:>5.1f}")

    for (batch, chunk), curve in sorted(curves.items()):
        print(f"\n--- batch {batch}, chunk {chunk} ---")
        total = 0.0
        for name, lengths in sorted(splits.items()):
            # Per-token cost is measured *at this batch size*, so concurrency is
            # already in the number; do not divide by batch again.
            seconds = sum(interpolate(curve, n) for n in lengths)
            total += seconds
            print(f"  {name:12s} n={len(lengths):3d} tokens={sum(lengths):>10,} "
                  f"compute={seconds / 3600:7.2f} h")
        print(f"  {'ALL':12s} n={len(all_lengths):3d} "
              f"tokens={sum(all_lengths):>10,} compute={total / 3600:7.2f} h")
        if len(curve) >= 3:
            a, b, worst = fit_linear_per_token(curve)
            fitted = sum(a * n + b * n * n for n in all_lengths)
            print(f"  fit c(L) = {a * 1e3:.4f} ms + {b * 1e9:.5f} ns*L per token "
                  f"(worst residual {worst * 100:.0f}%) -> {fitted / 3600:.2f} h")
        groups = -(-len(all_lengths) // batch)
        io_seconds = groups * ROUTED_BYTES_BACKBONE / 1e9 / opts.ssd_gb_per_s
        print(f"  layer-sequential I/O: {groups} passes x "
              f"{ROUTED_BYTES_BACKBONE / 1e9:.1f} GB @ {opts.ssd_gb_per_s} GB/s "
              f"= {io_seconds / 3600:.2f} h")
        print(f"  overlapped wall = max(compute, I/O) = "
              f"{max(total, io_seconds) / 3600:.2f} h")


if __name__ == "__main__":
    main()
