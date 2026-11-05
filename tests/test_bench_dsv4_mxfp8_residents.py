"""The mxfp8-resident kill test: its exactness half and its verdict rules."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from benchmarks import bench_dsv4_mxfp8_residents as bench  # noqa: E402


def _row(bf16: float, mxfp8: float, affine8: float) -> dict:
    return {
        "bf16": {"median_ms_per_call": bf16},
        "mxfp8": {"median_ms_per_call": mxfp8},
        "affine8_g64": {"median_ms_per_call": affine8},
    }


_EXACT = {"t": {"bit_mismatches": 0}}


def test_verdict_passes_only_when_exact_and_fast_on_a_gpu():
    timings = {"t": _row(bf16=2.0, mxfp8=1.0, affine8=1.0)}
    assert bench.verdict(_EXACT, timings, speed_valid=True)["verdict"] == "PASS"
    # Same numbers from a CPU run prove exactness only.
    assert bench.verdict(_EXACT, timings, speed_valid=False)["verdict"] == "EXACT_ONLY"
    assert bench.verdict(_EXACT, None, speed_valid=True)["verdict"] == "EXACT_ONLY"


@pytest.mark.parametrize(
    ("bf16", "mxfp8", "affine8", "needle"),
    [
        (1.4, 1.0, 1.0, "below the 1.5x bar"),
        (2.0, 1.2, 1.0, "affine-8"),
    ],
)
def test_verdict_kills_on_either_speed_criterion(bf16, mxfp8, affine8, needle):
    result = bench.verdict(_EXACT, {"t": _row(bf16, mxfp8, affine8)}, speed_valid=True)
    assert result["verdict"] == "KILL"
    assert any(needle in reason for reason in result["reasons"])


def test_verdict_kills_on_any_bit_mismatch_even_without_timings():
    result = bench.verdict({"t": {"bit_mismatches": 1}}, None, speed_valid=False)
    assert result["verdict"] == "KILL" and result["exact"] is False


def test_speed_is_judged_on_the_summed_decode_step_not_per_tensor():
    timings = {
        "fast": _row(bf16=3.0, mxfp8=1.0, affine8=1.0),
        "slow": _row(bf16=1.0, mxfp8=1.0, affine8=1.0),
    }
    result = bench.verdict(_EXACT, timings, speed_valid=True)
    assert result["mxfp8_speedup_over_bf16"] == pytest.approx(2.0)
    assert result["verdict"] == "PASS"


def test_synthetic_run_is_exact_and_reports_every_variant():
    result = bench.run(
        tensors=["a", "b"],
        layer=0,
        checkpoint=None,
        synthetic_shapes={"a": (64, 256), "b": (130, 192)},  # ragged row and column blocks
        copies=2,
        iterations=2,
        warmup=1,
        rounds=2,
        seed=0,
        exactness_only=False,
    )
    assert result["record_type"] == bench.RECORD_TYPE
    assert result["verdict"]["exact"] is True
    for tensor in ("a", "b"):
        assert set(result["timings"][tensor]) == {"bf16", "mxfp8", "affine8_g64"}
        assert result["exactness"][tensor]["mxfp8_bytes"] < result["exactness"][tensor]["bf16_bytes"]
    if mx.default_device() != mx.gpu:
        assert result["verdict"]["verdict"] == "EXACT_ONLY"


def _write_fp8_checkpoint(root: Path, tensors: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    header: dict[str, object] = {}
    blobs: list[bytes] = []
    offset = 0
    for prefix, (codes, scales) in sorted(tensors.items()):
        for suffix, dtype, array in (
            ("weight", "F8_E4M3", codes),
            ("scale", "F8_E8M0", scales),
        ):
            raw = array.tobytes()
            header[f"{prefix}.{suffix}"] = {
                "dtype": dtype,
                "shape": list(array.shape),
                "data_offsets": [offset, offset + len(raw)],
            }
            offset += len(raw)
            blobs.append(raw)
    raw_header = json.dumps(header, separators=(",", ":")).encode()
    raw_header += b" " * ((-len(raw_header)) % 8)
    shard = "model-00001-of-00001.safetensors"
    with (root / shard).open("wb") as handle:
        handle.write(struct.pack("<Q", len(raw_header)))
        handle.write(raw_header)
        for blob in blobs:
            handle.write(blob)
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard for name in header}})
    )


def test_checkpoint_run_reads_real_fp8_tensors(tmp_path):
    rng = np.random.default_rng(1)
    _write_fp8_checkpoint(
        tmp_path,
        {"layers.3.attn.wq_a": bench.synthetic_fp8_pair(rng, (96, 256))},
    )
    result = bench.run(
        tensors=["attn.wq_a"],
        layer=3,
        checkpoint=tmp_path,
        synthetic_shapes=None,
        copies=1,
        iterations=1,
        warmup=0,
        rounds=1,
        seed=0,
        exactness_only=True,
    )
    assert result["shapes"] == {"attn.wq_a": [96, 256]}
    assert result["exactness"]["attn.wq_a"]["bit_mismatches"] == 0
    assert result["timings"] == {}


def test_main_refuses_custom_wired_limits(monkeypatch):
    monkeypatch.setenv("GLM_MLX_WIRED_LIMIT_GB", "100")
    with pytest.raises(SystemExit):
        bench.main(["--synthetic", "--exactness-only", "--no-lock"])


def test_wo_a_is_timed_in_its_grouped_runtime_form():
    result = bench.run(
        tensors=["attn.wo_a"],
        layer=0,
        checkpoint=None,
        synthetic_shapes={"attn.wo_a": (256, 128)},  # 8 groups of 32 rows
        copies=1,
        iterations=1,
        warmup=0,
        rounds=1,
        seed=0,
        exactness_only=False,
    )
    assert bench.GROUPED_TENSORS["attn.wo_a"] == 8
    assert result["exactness"]["attn.wo_a"]["bit_mismatches"] == 0
    assert set(result["timings"]["attn.wo_a"]) == {"bf16", "mxfp8", "affine8_g64"}


def test_grouped_timing_rejects_rows_that_do_not_split():
    codes, scales = bench.synthetic_fp8_pair(np.random.default_rng(0), (100, 128))
    with pytest.raises(ValueError, match="groups"):
        bench.time_variants(
            codes, scales, copies=1, iterations=1, warmup=0, rounds=1, seed=0, groups=8
        )
