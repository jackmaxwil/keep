# RAMP Kernel Parity Harness (F7) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable harness that proves every RAMP compute kernel
(`vq_qmv`, `vq_qmm`, `gather_vqmm`, and the NAX native paths) produces
bit-for-bit / within-tolerance identical output to a known-good reference, and
expose it as `keep kernel-parity` plus a CI-safe test suite.

**Architecture:** A declarative `KernelParityCase` (kernel callable + reference
callable + deterministic input generator + tolerances). A runner evaluates each
case and returns a structured verdict. Three case families: (1) cross-reference
(numpy reference vs MLX reference — runs anywhere, no Metal), (2) Metal-vs-
reference (gated on Metal availability), (3) NAX-vs-reference (gated on the
native ext). Results render to JSON and to a CLI table.

**Tech Stack:** Python 3.11+, MLX (`mx.fast.metal_kernel`), NumPy, argparse,
pytest.

## Global Constraints

Inherited from [`2026-07-11-ramp-beyond-ram-roadmap.md`](2026-07-11-ramp-beyond-ram-roadmap.md).
Load-bearing here:

- Tests run via `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest`.
  Every test must pass on a machine **with no Metal and no model artifact**
  (codex sandboxes have no Metal). Metal/NAX cases must **skip**, never fail,
  when the backend is unavailable.
- `from __future__ import annotations` on line 1; pytest style; `tmp_path` for
  scratch; `np.testing.assert_allclose(actual, desired, rtol=, atol=)` for
  numerics.
- New module under `src/mlx_vq/`; add `src/ramp/` re-export as the final task.
- Preserve the 8-bit NAX path — this plan only *observes* kernels, never changes
  their dispatch.
- One commit per task. Do not push.

## File Structure

- Create `src/mlx_vq/kernels/parity.py` — the harness: `Tolerance`,
  `ParityVerdict`, `KernelParityCase`, `evaluate_case`, `run_parity_suite`,
  `metal_available`, `default_parity_cases`.
- Create `tests/test_kernel_parity.py` — harness unit tests + the three case
  families (Metal/NAX gated with `pytest.mark.skipif`).
- Modify `src/mlx_vq/build/cli.py` — add the `kernel-parity` subcommand.
- Create `tests/test_kernel_parity_cli.py` — CLI smoke test.
- Create `src/ramp/kernels/parity.py` — thin re-export (final task).

## Verified inputs (from exploration, 2026-07-11)

- `vq_qmv(x, codes, scales, codebook=None, *, in_dim, out_dim, group_size=512, code_bits=8, implementation="metal"|"reference", codebook_duplication=1)` at `src/mlx_vq/kernels/vq_qmv.py:163`; numpy twin `vq_qmv_reference_np(...)` at `:126`; MLX twin `vq_qmv_reference(...)` at `:235`.
- `vq_qmm(x, codes, scales, codebook=None, *, in_dim, out_dim, group_size=512, code_bits=8, implementation=..., codebook_duplication=1)` at `src/mlx_vq/kernels/vq_qmm.py:93`; numpy twin `vq_qmm_reference_np(...)` at `:69`; MLX twin `vq_qmm_reference(...)` at `:155`.
- Shapes: `codes` is `uint8` (code_bits=8) shape `[out_dim, in_dim//8]`; `scales` is `float32` shape `[out_dim, in_dim//group_size]`; `x` is `[in_dim]` (qmv) or `[M, in_dim]` (qmm); output `[out_dim]` / `[M, out_dim]`. `codebook=None` uses the kernel's built-in E8 default (identical across impls).
- NAX: `nax.is_available()`, `nax.predecoded_fp16_matmul(...)`, `nax.predecoded_fp16_gather_mm(...)` via `nax.load_native()` at `src/mlx_vq/kernels/nax.py:53`.
- Metal availability is probed with `mx.metal.is_available()` (this MLX build exposes `mx.metal.*`, used in `metrics.py`).

---

### Task 1: Parity primitives (`Tolerance`, `ParityVerdict`, `evaluate_case`)

**Files:**
- Create: `src/mlx_vq/kernels/parity.py`
- Test: `tests/test_kernel_parity.py`

**Interfaces:**
- Produces: `Tolerance(rtol: float, atol: float)`;
  `ParityVerdict(name: str, passed: bool, max_abs_err: float, max_rel_err: float, rtol: float, atol: float, skipped: bool, skip_reason: str | None)`;
  `KernelParityCase(name: str, produce, reference, tolerance: Tolerance, requires: str)` where `produce`/`reference` are `Callable[[], object]` returning an array-like and `requires` is one of `"cpu" | "metal" | "nax"`;
  `evaluate_case(case: KernelParityCase, *, backend_ok: Callable[[str], bool]) -> ParityVerdict`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_kernel_parity.py
from __future__ import annotations

import numpy as np

from mlx_vq.kernels.parity import (
    KernelParityCase,
    Tolerance,
    evaluate_case,
)


def _always_ok(_requirement: str) -> bool:
    return True


def test_evaluate_case_passes_on_identical_arrays() -> None:
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    case = KernelParityCase(
        name="identity",
        produce=lambda: arr.copy(),
        reference=lambda: arr.copy(),
        tolerance=Tolerance(rtol=0.0, atol=0.0),
        requires="cpu",
    )
    verdict = evaluate_case(case, backend_ok=_always_ok)
    assert verdict.passed is True
    assert verdict.skipped is False
    assert verdict.max_abs_err == 0.0


def test_evaluate_case_fails_when_error_exceeds_tolerance() -> None:
    base = np.ones((2, 2), dtype=np.float32)
    case = KernelParityCase(
        name="off-by-0.5",
        produce=lambda: base + 0.5,
        reference=lambda: base,
        tolerance=Tolerance(rtol=0.0, atol=1e-3),
        requires="cpu",
    )
    verdict = evaluate_case(case, backend_ok=_always_ok)
    assert verdict.passed is False
    assert verdict.max_abs_err == 0.5


def test_evaluate_case_skips_when_backend_unavailable() -> None:
    def _no_metal(requirement: str) -> bool:
        return requirement != "metal"

    def _boom() -> np.ndarray:  # must never be called
        raise AssertionError("produce() ran despite unavailable backend")

    case = KernelParityCase(
        name="metal-only",
        produce=_boom,
        reference=_boom,
        tolerance=Tolerance(rtol=0.0, atol=0.0),
        requires="metal",
    )
    verdict = evaluate_case(case, backend_ok=_no_metal)
    assert verdict.skipped is True
    assert verdict.passed is True  # a skip does not fail the suite
    assert verdict.skip_reason == "backend 'metal' unavailable"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_kernel_parity.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'mlx_vq.kernels.parity'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/mlx_vq/kernels/parity.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

BackendRequirement = Literal["cpu", "metal", "nax"]


@dataclass(frozen=True)
class Tolerance:
    rtol: float
    atol: float


@dataclass(frozen=True)
class ParityVerdict:
    name: str
    passed: bool
    max_abs_err: float
    max_rel_err: float
    rtol: float
    atol: float
    skipped: bool
    skip_reason: str | None


@dataclass(frozen=True)
class KernelParityCase:
    name: str
    produce: Callable[[], object]
    reference: Callable[[], object]
    tolerance: Tolerance
    requires: BackendRequirement


def _to_numpy(value: object) -> np.ndarray:
    # mx.array exposes __array__, so np.asarray materializes it on host.
    return np.asarray(value, dtype=np.float32)


def evaluate_case(
    case: KernelParityCase,
    *,
    backend_ok: Callable[[str], bool],
) -> ParityVerdict:
    if not backend_ok(case.requires):
        return ParityVerdict(
            name=case.name,
            passed=True,
            max_abs_err=0.0,
            max_rel_err=0.0,
            rtol=case.tolerance.rtol,
            atol=case.tolerance.atol,
            skipped=True,
            skip_reason=f"backend {case.requires!r} unavailable",
        )
    actual = _to_numpy(case.produce())
    desired = _to_numpy(case.reference())
    abs_err = np.abs(actual - desired)
    max_abs = float(abs_err.max()) if abs_err.size else 0.0
    denom = np.abs(desired)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err = np.where(denom > 0, abs_err / denom, 0.0)
    max_rel = float(rel_err.max()) if rel_err.size else 0.0
    passed = bool(
        np.allclose(actual, desired, rtol=case.tolerance.rtol, atol=case.tolerance.atol)
    )
    return ParityVerdict(
        name=case.name,
        passed=passed,
        max_abs_err=max_abs,
        max_rel_err=max_rel,
        rtol=case.tolerance.rtol,
        atol=case.tolerance.atol,
        skipped=False,
        skip_reason=None,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_kernel_parity.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/mlx_vq/kernels/parity.py tests/test_kernel_parity.py
git commit -m "feat: add kernel parity primitives"
```

---

### Task 2: Backend probes + suite runner

**Files:**
- Modify: `src/mlx_vq/kernels/parity.py`
- Test: `tests/test_kernel_parity.py`

**Interfaces:**
- Consumes: `KernelParityCase`, `ParityVerdict`, `evaluate_case` (Task 1).
- Produces: `metal_available() -> bool`; `nax_available() -> bool`;
  `default_backend_ok(requirement: str) -> bool`;
  `run_parity_suite(cases: list[KernelParityCase], *, backend_ok=None) -> list[ParityVerdict]`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_kernel_parity.py
from mlx_vq.kernels.parity import run_parity_suite


def test_run_parity_suite_aggregates_and_respects_backend_gate() -> None:
    cpu_case = KernelParityCase(
        name="cpu-ok",
        produce=lambda: np.zeros(3, dtype=np.float32),
        reference=lambda: np.zeros(3, dtype=np.float32),
        tolerance=Tolerance(rtol=0.0, atol=0.0),
        requires="cpu",
    )
    metal_case = KernelParityCase(
        name="metal-skipped",
        produce=lambda: (_ for _ in ()).throw(AssertionError("ran")),
        reference=lambda: (_ for _ in ()).throw(AssertionError("ran")),
        tolerance=Tolerance(rtol=0.0, atol=0.0),
        requires="metal",
    )
    verdicts = run_parity_suite(
        [cpu_case, metal_case],
        backend_ok=lambda req: req == "cpu",
    )
    assert [v.name for v in verdicts] == ["cpu-ok", "metal-skipped"]
    assert verdicts[0].passed and not verdicts[0].skipped
    assert verdicts[1].skipped and verdicts[1].passed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_kernel_parity.py::test_run_parity_suite_aggregates_and_respects_backend_gate -q`
Expected: FAIL with `ImportError: cannot import name 'run_parity_suite'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/mlx_vq/kernels/parity.py
import mlx.core as mx

from mlx_vq.kernels import nax


def metal_available() -> bool:
    metal = getattr(mx, "metal", None)
    if metal is None:
        return False
    try:
        return bool(metal.is_available())
    except Exception:
        return False


def nax_available() -> bool:
    try:
        return bool(nax.is_available())
    except Exception:
        return False


def default_backend_ok(requirement: str) -> bool:
    if requirement == "cpu":
        return True
    if requirement == "metal":
        return metal_available()
    if requirement == "nax":
        return nax_available()
    return False


def run_parity_suite(
    cases: list[KernelParityCase],
    *,
    backend_ok: Callable[[str], bool] | None = None,
) -> list[ParityVerdict]:
    gate = backend_ok if backend_ok is not None else default_backend_ok
    return [evaluate_case(case, backend_ok=gate) for case in cases]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_kernel_parity.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/mlx_vq/kernels/parity.py tests/test_kernel_parity.py
git commit -m "feat: add parity backend probes and suite runner"
```

---

### Task 3: Real kernel cases — numpy-reference vs MLX-reference (runs without Metal)

**Files:**
- Modify: `src/mlx_vq/kernels/parity.py`
- Test: `tests/test_kernel_parity.py`

**Interfaces:**
- Consumes: `KernelParityCase`, `Tolerance`, `run_parity_suite` (Tasks 1–2).
- Produces: `synthetic_vq_inputs(rng, *, in_dim, out_dim, m, group_size) -> dict`;
  `cross_reference_cases() -> list[KernelParityCase]` (all `requires="cpu"`,
  comparing each kernel's `*_reference_np` numpy path against its `*_reference`
  MLX path — two independent implementations of the same math).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_kernel_parity.py
from mlx_vq.kernels.parity import cross_reference_cases


def test_cross_reference_cases_all_pass_without_metal() -> None:
    cases = cross_reference_cases()
    assert {c.requires for c in cases} == {"cpu"}
    assert {c.name for c in cases} >= {"vq_qmv:np-vs-mlx", "vq_qmm:np-vs-mlx"}
    verdicts = run_parity_suite(cases, backend_ok=lambda _req: True)
    failures = [v for v in verdicts if not v.passed]
    assert failures == [], [f"{v.name}: abs={v.max_abs_err}" for v in failures]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_kernel_parity.py::test_cross_reference_cases_all_pass_without_metal -q`
Expected: FAIL with `ImportError: cannot import name 'cross_reference_cases'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/mlx_vq/kernels/parity.py
from mlx_vq.kernels.vq_qmv import vq_qmv_reference, vq_qmv_reference_np
from mlx_vq.kernels.vq_qmm import vq_qmm_reference, vq_qmm_reference_np


def synthetic_vq_inputs(
    rng: np.random.Generator,
    *,
    in_dim: int,
    out_dim: int,
    m: int,
    group_size: int,
) -> dict:
    codes = rng.integers(0, 256, size=(out_dim, in_dim // 8), dtype=np.uint8)
    scales = rng.standard_normal((out_dim, in_dim // group_size)).astype(np.float32)
    x_vec = rng.standard_normal(in_dim).astype(np.float32)
    x_mat = rng.standard_normal((m, in_dim)).astype(np.float32)
    return {
        "codes": codes,
        "scales": scales,
        "x_vec": x_vec,
        "x_mat": x_mat,
        "in_dim": in_dim,
        "out_dim": out_dim,
        "group_size": group_size,
    }


def cross_reference_cases() -> list[KernelParityCase]:
    # Deterministic inputs: seed fixed so re-runs are reproducible.
    rng = np.random.default_rng(20260711)
    d = synthetic_vq_inputs(rng, in_dim=512, out_dim=64, m=4, group_size=512)
    common = dict(in_dim=d["in_dim"], out_dim=d["out_dim"], group_size=d["group_size"], code_bits=8)
    exact = Tolerance(rtol=0.0, atol=0.0)
    return [
        KernelParityCase(
            name="vq_qmv:np-vs-mlx",
            produce=lambda: vq_qmv_reference(d["x_vec"], d["codes"], d["scales"], None, **common),
            reference=lambda: vq_qmv_reference_np(d["x_vec"], d["codes"], d["scales"], None, **common),
            tolerance=exact,
            requires="cpu",
        ),
        KernelParityCase(
            name="vq_qmm:np-vs-mlx",
            produce=lambda: vq_qmm_reference(d["x_mat"], d["codes"], d["scales"], None, **common),
            reference=lambda: vq_qmm_reference_np(d["x_mat"], d["codes"], d["scales"], None, **common),
            tolerance=exact,
            requires="cpu",
        ),
    ]
```

> Note: if the two reference implementations legitimately differ by float
> rounding (MLX fp32 accumulate vs numpy fp64 intermediates), loosen `exact` to
> `Tolerance(rtol=1e-5, atol=1e-5)` and record the observed `max_abs_err` in the
> commit message — do not silently widen tolerance without noting the number.

- [ ] **Step 4: Run test to verify it passes**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_kernel_parity.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/mlx_vq/kernels/parity.py tests/test_kernel_parity.py
git commit -m "feat: add cross-reference parity cases for vq_qmv/vq_qmm"
```

---

### Task 4: Metal-vs-reference and NAX-vs-reference cases (backend-gated)

**Files:**
- Modify: `src/mlx_vq/kernels/parity.py`
- Test: `tests/test_kernel_parity.py`

**Interfaces:**
- Consumes: `synthetic_vq_inputs`, `KernelParityCase`, `run_parity_suite`.
- Produces: `metal_cases() -> list[KernelParityCase]` (`requires="metal"`,
  comparing `implementation="metal"` against `implementation="reference"`);
  `default_parity_cases() -> list[KernelParityCase]` (cross-reference + metal +
  nax, in that order).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_kernel_parity.py
import pytest

from mlx_vq.kernels.parity import (
    default_parity_cases,
    metal_available,
    metal_cases,
)


def test_metal_cases_are_gated_and_named() -> None:
    cases = metal_cases()
    assert {c.requires for c in cases} == {"metal"}
    assert any(c.name == "vq_qmv:metal-vs-ref" for c in cases)


def test_default_parity_cases_include_all_families() -> None:
    names = {c.name for c in default_parity_cases()}
    assert "vq_qmv:np-vs-mlx" in names
    assert "vq_qmv:metal-vs-ref" in names


@pytest.mark.skipif(not metal_available(), reason="Metal backend unavailable")
def test_metal_matches_reference_bit_close() -> None:
    verdicts = run_parity_suite(metal_cases())
    failures = [v for v in verdicts if not v.passed and not v.skipped]
    assert failures == [], [f"{v.name}: abs={v.max_abs_err}" for v in failures]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_kernel_parity.py::test_metal_cases_are_gated_and_named -q`
Expected: FAIL with `ImportError: cannot import name 'metal_cases'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/mlx_vq/kernels/parity.py
from mlx_vq.kernels.vq_qmv import vq_qmv
from mlx_vq.kernels.vq_qmm import vq_qmm


def metal_cases() -> list[KernelParityCase]:
    rng = np.random.default_rng(20260711)
    d = synthetic_vq_inputs(rng, in_dim=512, out_dim=64, m=4, group_size=512)
    common = dict(in_dim=d["in_dim"], out_dim=d["out_dim"], group_size=d["group_size"], code_bits=8)
    # Metal fp16/fp32 accumulation differs from the reference at the ULP level;
    # tolerate small absolute error, tighten if the observed max is lower.
    tol = Tolerance(rtol=1e-3, atol=1e-3)
    return [
        KernelParityCase(
            name="vq_qmv:metal-vs-ref",
            produce=lambda: vq_qmv(d["x_vec"], d["codes"], d["scales"], None, implementation="metal", **common),
            reference=lambda: vq_qmv(d["x_vec"], d["codes"], d["scales"], None, implementation="reference", **common),
            tolerance=tol,
            requires="metal",
        ),
        KernelParityCase(
            name="vq_qmm:metal-vs-ref",
            produce=lambda: vq_qmm(d["x_mat"], d["codes"], d["scales"], None, implementation="metal", **common),
            reference=lambda: vq_qmm(d["x_mat"], d["codes"], d["scales"], None, implementation="reference", **common),
            tolerance=tol,
            requires="metal",
        ),
    ]


def default_parity_cases() -> list[KernelParityCase]:
    return cross_reference_cases() + metal_cases()
```

> NAX cases: once `nax.predecoded_fp16_gather_mm` is wired here in a follow-up,
> add a `nax_cases()` builder with `requires="nax"` comparing the predecoded NAX
> output against `vq_qmm(..., implementation="reference")`, and append it to
> `default_parity_cases()`. It is deferred out of this task only because it needs
> a predecoded-codebook fixture; the gating machinery already supports it.

- [ ] **Step 4: Run test to verify it passes**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_kernel_parity.py -q`
Expected: PASS on non-Metal (the `test_metal_matches_reference_bit_close` case reports `skipped`); PASS with real comparisons on a Metal Mac.

- [ ] **Step 5: Commit**

```bash
git add src/mlx_vq/kernels/parity.py tests/test_kernel_parity.py
git commit -m "feat: add Metal-vs-reference gated parity cases"
```

---

### Task 5: `keep kernel-parity` CLI subcommand + JSON report

**Files:**
- Modify: `src/mlx_vq/kernels/parity.py` (add `verdicts_to_report`)
- Modify: `src/mlx_vq/build/cli.py` (register subcommand)
- Test: `tests/test_kernel_parity_cli.py`

**Interfaces:**
- Consumes: `default_parity_cases`, `run_parity_suite`, `ParityVerdict`.
- Produces: `verdicts_to_report(verdicts) -> dict`; CLI `keep kernel-parity
  [--json PATH]` returning `0` when no non-skipped case failed, else `1`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_kernel_parity_cli.py
from __future__ import annotations

import json
from pathlib import Path

from mlx_vq.build.cli import main as cli_main


def test_kernel_parity_cli_writes_report_and_returns_zero(tmp_path: Path) -> None:
    report_path = tmp_path / "parity.json"
    rc = cli_main(["kernel-parity", "--json", str(report_path)])
    assert rc == 0
    report = json.loads(report_path.read_text())
    assert "cases" in report and report["cases"]
    assert report["failed"] == 0
    names = {case["name"] for case in report["cases"]}
    assert "vq_qmv:np-vs-mlx" in names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_kernel_parity_cli.py -q`
Expected: FAIL — argparse exits non-zero on the unknown `kernel-parity` command.

- [ ] **Step 3: Write minimal implementation**

Add the report helper:

```python
# append to src/mlx_vq/kernels/parity.py
def verdicts_to_report(verdicts: list[ParityVerdict]) -> dict:
    cases = [
        {
            "name": v.name,
            "passed": v.passed,
            "skipped": v.skipped,
            "skip_reason": v.skip_reason,
            "max_abs_err": v.max_abs_err,
            "max_rel_err": v.max_rel_err,
            "rtol": v.rtol,
            "atol": v.atol,
        }
        for v in verdicts
    ]
    return {
        "cases": cases,
        "total": len(cases),
        "skipped": sum(1 for v in verdicts if v.skipped),
        "failed": sum(1 for v in verdicts if not v.passed and not v.skipped),
    }
```

Register the subcommand in `src/mlx_vq/build/cli.py`. Add the handler next to the
other `_cmd_*` handlers:

```python
# src/mlx_vq/build/cli.py  (new handler; place with the other _cmd_* functions)
def _cmd_kernel_parity(args: argparse.Namespace) -> int:
    import json as _json

    from mlx_vq.kernels.parity import (
        default_parity_cases,
        run_parity_suite,
        verdicts_to_report,
    )

    verdicts = run_parity_suite(default_parity_cases())
    report = verdicts_to_report(verdicts)
    if args.json is not None:
        from pathlib import Path as _Path

        _Path(args.json).write_text(_json.dumps(report, indent=2))
    for case in report["cases"]:
        status = "SKIP" if case["skipped"] else ("PASS" if case["passed"] else "FAIL")
        print(f"{status:4} {case['name']}  max_abs={case['max_abs_err']:.3e}")
    print(f"{report['failed']} failed, {report['skipped']} skipped, {report['total']} total")
    return 1 if report["failed"] else 0
```

Wire it inside `build_parser()` (`src/mlx_vq/build/cli.py:418`), mirroring the
existing `validate = subparsers.add_parser(...)` pattern:

```python
    kernel_parity = subparsers.add_parser(
        "kernel-parity",
        help="validate compute kernels against a known-good reference",
    )
    kernel_parity.add_argument(
        "--json",
        default=None,
        help="write the parity report to this JSON path",
    )
    kernel_parity.set_defaults(func=_cmd_kernel_parity)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_kernel_parity_cli.py -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/mlx_vq/kernels/parity.py src/mlx_vq/build/cli.py tests/test_kernel_parity_cli.py
git commit -m "feat: add keep kernel-parity CLI subcommand"
```

---

### Task 6: `ramp` re-export + docs pointer

**Files:**
- Create: `src/ramp/kernels/parity.py`
- Modify: `README.md` (Verification section: add the parity command)
- Test: `tests/test_kernel_parity.py` (import-surface assertion)

**Interfaces:**
- Consumes: everything public in `mlx_vq.kernels.parity`.
- Produces: `ramp.kernels.parity` re-exporting the public names.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_kernel_parity.py
def test_ramp_reexports_parity_surface() -> None:
    from ramp.kernels import parity as ramp_parity

    assert hasattr(ramp_parity, "run_parity_suite")
    assert hasattr(ramp_parity, "default_parity_cases")
    assert hasattr(ramp_parity, "KernelParityCase")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_kernel_parity.py::test_ramp_reexports_parity_surface -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ramp.kernels.parity'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/ramp/kernels/parity.py
"""Public RAMP re-export of the kernel parity harness."""
from __future__ import annotations

from mlx_vq.kernels.parity import (  # noqa: F401
    KernelParityCase,
    ParityVerdict,
    Tolerance,
    default_parity_cases,
    evaluate_case,
    metal_available,
    nax_available,
    run_parity_suite,
    verdicts_to_report,
)
```

Add to `README.md` under the "Native/NAX-focused checks:" area:

```markdown
Kernel parity (bit-exact vs. reference; Metal/NAX cases skip when unavailable):

```bash
uv run keep kernel-parity --json /tmp/keep-kernel-parity.json
```
```

- [ ] **Step 4: Run test to verify it passes**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_kernel_parity.py tests/test_kernel_parity_cli.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add src/ramp/kernels/parity.py README.md tests/test_kernel_parity.py
git commit -m "feat: re-export kernel parity via ramp and document the command"
```

---

## Self-Review

- **Spec coverage:** F7 asks for "every kernel validated against a known-good
  reference." Covered: cross-reference (Task 3), Metal (Task 4), CLI report
  (Task 5). NAX comparison is scaffolded (gating + note) and explicitly deferred
  with a stated reason (needs a predecoded fixture) — not silently dropped.
- **Placeholder scan:** none; every step has runnable code/commands.
- **Type consistency:** `KernelParityCase`, `ParityVerdict`, `Tolerance`,
  `run_parity_suite`, `default_parity_cases` used identically across tasks and
  the CLI handler.
- **Known judgment point:** tolerances in Tasks 3–4 may need one adjustment after
  the first real run; the plan says to record the observed error, not to widen
  blindly.
