#!/usr/bin/env bash
# Phase 0 kill tests that need no model download. Run on the target Mac:
#
#   scripts/phase0_kill_tests.sh            # writes artifacts/benchmarks/phase0-<host>-<date>/
#
# Each benchmark exits 1 on KILL. The script runs them all, then prints one
# verdict line per test. Optional real-weight variants, if the files exist:
#   KEEP_RESIDENT_DIR=~/keep-artifacts/dsv4-residents-release-precision-20260819
#   KEEP_VQ_DIR=~/keep-artifacts/dsv4-vq-e8p-g512
set -uo pipefail

for name in GLM_MLX_WIRED_LIMIT_GB GLM_SINGLE_HOST_MLX_WIRED_LIMIT_GB; do
  if [[ -n "${!name:-}" ]]; then
    echo "unset $name first: kill tests run under the default wired-memory limits" >&2
    exit 2
  fi
done

repo="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo"
out="artifacts/benchmarks/phase0-$(hostname -s)-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$out"
status=0

run() {
  local label="$1"; shift
  echo "== $label"
  if uv run python "$@"; then echo "$label: ok"; else echo "$label: KILL or error (exit $?)"; status=1; fi
}

run "mxfp8 residents (synthetic V4-Flash shapes)" \
  benchmarks/bench_dsv4_mxfp8_residents.py --synthetic --output-json "$out/mxfp8-residents-synthetic.json"
if [[ -n "${KEEP_RESIDENT_DIR:-}" && -d "${KEEP_RESIDENT_DIR}" ]]; then
  run "mxfp8 residents (real layer 20)" \
    benchmarks/bench_dsv4_mxfp8_residents.py --checkpoint "$KEEP_RESIDENT_DIR" --layer 20 \
    --output-json "$out/mxfp8-residents-layer20.json"
fi
run "fused hyper-connection sinkhorn" \
  benchmarks/bench_hc_sinkhorn.py --output-json "$out/hc-sinkhorn.json"
run "E8P single-token kernels (synthetic)" \
  benchmarks/bench_e8p_m1.py --output-json "$out/e8p-m1-synthetic.json"
if [[ -n "${KEEP_VQ_DIR:-}" && -d "${KEEP_VQ_DIR}" ]]; then
  run "E8P single-token kernels (real layer 20)" \
    benchmarks/bench_e8p_m1.py --artifact-dir "$KEEP_VQ_DIR" --layer 20 \
    --output-json "$out/e8p-m1-layer20.json"
fi

echo "== Metal parity tests"
if uv run --group dev python -m pytest -q \
    tests/test_hc_sinkhorn.py tests/test_e8p_m1.py tests/test_fp8_block_dequant.py \
    tests/test_bench_dsv4_mxfp8_residents.py "tests/test_deepseek_v4_flash_adapter.py" -k "mxfp8 or sinkhorn or e8p or fp8" \
    | tee "$out/metal-parity-tests.txt"; then
  echo "Metal parity tests: ok"
else
  echo "Metal parity tests: FAILED"; status=1
fi

echo
echo "Verdicts:"
for f in "$out"/*.json; do
  uv run python - "$f" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
v = d.get("verdict")
v = v.get("verdict") if isinstance(v, dict) else v
print(f"  {sys.argv[1].rsplit('/', 1)[-1]}: {v}")
PY
done
echo "Evidence: $out  (force-add it: git add -f $out)"
exit $status
