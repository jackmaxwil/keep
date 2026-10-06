"""Block-local PV-Tuning kill-gate spike for GLM-4.5-Air VQ-1.0 (frozen E8 codebook).

This is the milestone-3 "kill gate" from docs/architecture/pv-tuning-architecture.md.
It alternates, on ONE hard MoE layer's gate/up/down projections:

  P-step (continuous):  tune per-group fp16 scales via the differentiable surrogate's
                        closed-form least-squares fit (fit_scale_delta_sidecar_least_squares)
                        against the fp16 teacher projection output. With a FROZEN E8
                        1-bit codebook the only continuous lever is the per-group scale,
                        exactly as the architecture doc predicts.
  V-step (discrete):    reassign the E8 codes to nearest codeword in the P-updated scaled
                        space via hessian_weighted_reassign_codes (diagonal-Hessian proxy).

It reuses existing machinery verbatim:
  - P-step  -> mlx_vq.quality.mlx_surrogate.fit_scale_delta_sidecar_least_squares
  - V-step  -> mlx_vq.quality.hessian_rounding.imatrix_weighted_reassign_codes
               (-> hessian_weighted_reassign_codes)

Block-local metric (the kill gate): the projection-output distribution KLD of teacher
vs candidate on the real routed input states, plus rel-L2 weight error and the
hessian-weighted reconstruction error the V-step already reports. We compare four arms:
RTN baseline, P-only, V-only, and PV (alternating P+V for K rounds).

This is a SPIKE for a GO/NO-GO verdict; it is block-local and bounded. It does not run a
full-model forward, does not export an artifact, and does not touch the fused Metal kernel.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_vq.codebook.e8 import CODEWORD_DIM, decode_e8_1bit
from mlx_vq.convert.inspect_hf import GLM45_AIR_MODEL_ID, summarize_config
from mlx_vq.convert.stream_convert import load_safetensors_index
from mlx_vq.models.glm4_moe_adapter import GLM4MoEGate
from mlx_vq.quality.hessian_rounding import imatrix_weighted_reassign_codes
from mlx_vq.quality.mlx_surrogate import (
    SwitchLinearSurrogate,
    fit_scale_delta_sidecar_least_squares,
)
from mlx_vq.validate.glm45_air_hessian_probe import _decode_from_full_codebook
from mlx_vq.validate.glm45_air_vq import (
    _expert_weight_name,
    _load_switch_glu,
    _read_named_tensor,
    _routing_config,
)

GLU_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def _resolve_source_dir(source_dir: str | None) -> Path:
    if source_dir is not None:
        return Path(source_dir)
    import glob

    hits = glob.glob(
        str(Path.home() / ".cache/huggingface/hub/models--zai-org--GLM-4.5-Air/snapshots/*/")
    )
    if not hits:
        raise FileNotFoundError("GLM-4.5-Air HF checkpoint not found in HF cache; pass --source-dir")
    return Path(hits[0])


# ---------------------------------------------------------------------------
# Block-local calibration: real routed MoE-input states for one layer.
# ---------------------------------------------------------------------------
def _capture_moe_input_states(
    *,
    source_dir: Path,
    config_path: Path,
    index_path: Path,
    layer: int,
    prompts: list[str],
    max_tokens_per_prompt: int,
) -> np.ndarray:
    """Run a bounded resident-Air prefill and capture post_attention_layernorm output
    (the MoE input) at the target layer for real routed hidden states."""
    from mlx_vq.benchmark.glm45_air import load_resident_air
    from mlx_vq.models.glm45_air_vq_adapter import GLM45AirVQMoE

    model, tokenizer, _, _ = load_resident_air(
        source_dir=str(source_dir),
        config_path=str(config_path),
        index_path=str(index_path),
        artifact_dir="artifacts/glm-4.5-air-vq",
    )
    layer_module = model.model.layers[layer]
    if not isinstance(layer_module.mlp, GLM45AirVQMoE):
        raise ValueError(f"layer {layer} is not a sparse GLM-4.5-Air MoE layer")

    captured: list[np.ndarray] = []
    original_mlp = layer_module.mlp

    class _Tap:
        def __init__(self, inner):
            self.inner = inner

        def __call__(self, x):
            captured.append(np.asarray(x.reshape((-1, x.shape[-1])).astype(mx.float32)))
            return self.inner(x)

        def __getattr__(self, name):
            return getattr(self.inner, name)

    layer_module.mlp = _Tap(original_mlp)
    try:
        for prompt in prompts:
            token_ids = list(tokenizer.encode(prompt, add_special_tokens=False))[:max_tokens_per_prompt]
            if not token_ids:
                continue
            logits = model(mx.array([token_ids], dtype=mx.int32))
            mx.eval(logits)
    finally:
        layer_module.mlp = original_mlp
    if not captured:
        raise RuntimeError("captured no MoE input states")
    return np.concatenate(captured, axis=0).astype(np.float32)


def _route_states(config, source_dir, index, layer, x, model_id):
    """Return per-token top-k routing indices, shape [tokens, top_k]."""
    routing = _routing_config(config, model_id=model_id)
    gate_weight = _read_named_tensor(source_dir, index, f"model.layers.{layer}.mlp.gate.weight")
    correction_name = f"model.layers.{layer}.mlp.gate.e_score_correction_bias"
    if correction_name in index.weight_map:
        correction = _read_named_tensor(source_dir, index, correction_name)
    else:
        correction = np.zeros((summarize_config(config, model_id=model_id).n_routed_experts,), dtype=np.float32)
    gate = GLM4MoEGate(routing, weight=mx.array(gate_weight), e_score_correction_bias=mx.array(correction))
    indices_mx, _ = gate(mx.array(x))
    mx.eval(indices_mx)
    return np.asarray(indices_mx, dtype=np.int64)


# ---------------------------------------------------------------------------
# Block-local metric.
# ---------------------------------------------------------------------------
def _apply_signs(layer, x_np):
    signs = layer.get("rht_signs")
    if signs is None:
        return x_np
    return np.asarray((mx.array(x_np) * signs).astype(mx.float32), dtype=np.float32)


def _projection_output(layer, x_np, indices_np, *, effective_scales_np, codes_np, table):
    """Decode weights (codes+scales, frozen E8 table) and apply per routed slot.

    Returns [tokens, top_k, out] to match the surrogate's per-token-topk layout.
    RHT is off for the seed artifact (no rht_signs), so input space == weight space.
    """
    x_in = _apply_signs(layer, x_np)
    tokens, top_k = indices_np.shape
    out = np.zeros((tokens, top_k, layer.output_dims), dtype=np.float32)
    decoded_cache: dict[int, np.ndarray] = {}
    for expert in np.unique(indices_np):
        decoded_cache[int(expert)] = _decode_from_full_codebook(
            codes_np[int(expert)], effective_scales_np[int(expert)], table, group_size=layer.group_size
        )
    for t in range(tokens):
        for k in range(top_k):
            w = decoded_cache[int(indices_np[t, k])]
            out[t, k] = w @ x_in[t]
    return out


def _teacher_output(source_dir, index, layer_no, layer, projection, x_np, indices_np):
    """Teacher fp16 projection output, shape [tokens, top_k, out]."""
    x_in = _apply_signs(layer, x_np)
    tokens, top_k = indices_np.shape
    out = np.zeros((tokens, top_k, layer.output_dims), dtype=np.float32)
    weight_cache: dict[int, np.ndarray] = {}
    for expert in np.unique(indices_np):
        weight_cache[int(expert)] = _read_named_tensor(
            source_dir, index, _expert_weight_name(layer_no, int(expert), projection)
        ).astype(np.float32)
    for t in range(tokens):
        for k in range(top_k):
            out[t, k] = weight_cache[int(indices_np[t, k])] @ x_in[t]
    return out


def _row_softmax_kld(teacher: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    """Block-local output KLD: treat each routed projection-output row as a logit vector,
    softmax it, and report mean & p99.9 KL(teacher||candidate). Also rel-L2 of raw outputs.

    Inputs are [tokens, top_k, out]; flattened to [routes, out]."""
    out_dim = teacher.shape[-1]
    teacher = teacher.reshape((-1, out_dim))
    candidate = candidate.reshape((-1, out_dim))
    t = teacher.astype(np.float64)
    c = candidate.astype(np.float64)
    t_lse = t - (t.max(axis=-1, keepdims=True) + np.log(np.exp(t - t.max(axis=-1, keepdims=True)).sum(axis=-1, keepdims=True)))
    c_lse = c - (c.max(axis=-1, keepdims=True) + np.log(np.exp(c - c.max(axis=-1, keepdims=True)).sum(axis=-1, keepdims=True)))
    tp = np.exp(t_lse)
    kld = np.sum(tp * (t_lse - c_lse), axis=-1)
    diff = candidate.astype(np.float64) - teacher.astype(np.float64)
    rel_l2 = float(np.linalg.norm(diff.reshape(-1)) / max(np.linalg.norm(teacher.astype(np.float64).reshape(-1)), 1e-12))
    return {
        "mean_kld": float(np.mean(kld)),
        "p999_kld": float(np.percentile(kld, 99.9)) if kld.size else 0.0,
        "max_kld": float(np.max(kld)) if kld.size else 0.0,
        "rel_l2": rel_l2,
        "routes": int(teacher.shape[0]),
    }


# ---------------------------------------------------------------------------
# P-step and V-step, block-local, reusing existing machinery.
# ---------------------------------------------------------------------------
def _p_step(layer, x_mx, indices_mx, target_mx, *, ridge_strength: float, max_abs_log_delta: float) -> np.ndarray:
    """Continuous P-step: closed-form least-squares scale tuning through the surrogate.
    Returns the new scale_delta (log-multiplier) [experts, out, groups]."""
    result = fit_scale_delta_sidecar_least_squares(
        layer, x_mx, indices_mx, target_mx,
        ridge_strength=ridge_strength, max_abs_log_delta=max_abs_log_delta,
    )
    sidecar = result["sidecar"]
    return np.asarray(sidecar.scale_delta, dtype=np.float32), result


def _v_step(layer_no, layer, projection, source_dir, index, *, codes_np, effective_scales_np, table, hessian_diag, selected_experts):
    """Discrete V-step: reassign codes to nearest E8 codeword in the P-updated scaled space."""
    new_codes = codes_np.copy()
    changed = 0
    total = 0
    for expert in selected_experts:
        w = _read_named_tensor(source_dir, index, _expert_weight_name(layer_no, expert, projection)).astype(np.float32)
        reassigned = imatrix_weighted_reassign_codes(
            w,
            codes_np[expert],
            effective_scales_np[expert],
            hessian_diag,
            codebook=table,
            group_size=layer.group_size,
            code_bits=layer.code_bits,
        )
        new_codes[expert] = reassigned.codes
        changed += reassigned.stats.changed_code_count
        total += reassigned.stats.code_count
    return new_codes, changed, total


def _run_projection(
    *, layer_no, projection, switch_glu, source_dir, index, x_np, indices_np, rounds, ridge_strength, max_abs_log_delta
):
    layer = getattr(switch_glu, projection)
    if layer.code_bits != 8:
        raise ValueError(f"L{layer_no} {projection} code_bits={layer.code_bits}; spike supports frozen 8-bit E8 only")
    table = decode_e8_1bit(np.arange(256, dtype=np.uint8), np.asarray(layer.codebook))

    selected_experts = sorted(set(int(e) for e in np.unique(indices_np)))
    base_scales_np = np.asarray(layer.scales, dtype=np.float32)
    seed_codes_np = np.asarray(layer.codes).copy()

    # RHT-transformed diagonal hessian in the space the V-step decodes (matches probe convention).
    signs = layer.get("rht_signs")
    x_for_h = x_np if signs is None else np.asarray((mx.array(x_np) * signs).astype(mx.float32), dtype=np.float32)
    hessian_diag = np.mean(x_for_h.astype(np.float64) * x_for_h.astype(np.float64), axis=0).astype(np.float32)

    x_mx = mx.array(x_np)
    indices_mx = mx.array(indices_np)
    teacher = _teacher_output(source_dir, index, layer_no, layer, projection, x_np, indices_np)
    target_mx = mx.array(teacher)

    def metric_for(codes_np, effective_scales_np):
        cand = _projection_output(
            layer, x_np, indices_np, effective_scales_np=effective_scales_np, codes_np=codes_np, table=table
        )
        return _row_softmax_kld(teacher, cand)

    arms: dict[str, dict] = {}

    # --- Arm 1: RTN baseline (seed codes, seed scales, no PV) ---
    arms["rtn"] = {"metric": metric_for(seed_codes_np, base_scales_np), "changed_code_fraction": 0.0}

    # --- Arm 2: P-only (one closed-form scale fit against teacher, seed codes) ---
    layer.set_continuous_sidecar(scale_delta=None, output_bias=None)
    scale_delta_np, p_res = _p_step(layer, x_mx, indices_mx, target_mx, ridge_strength=ridge_strength, max_abs_log_delta=max_abs_log_delta)
    p_eff = base_scales_np * np.exp(scale_delta_np)
    arms["p_only"] = {"metric": metric_for(seed_codes_np, p_eff), "improvement_ratio": float(p_res["improvement_ratio"])}
    layer.set_continuous_sidecar(scale_delta=None, output_bias=None)

    # --- Arm 3: V-only (one reassignment against teacher in seed-scale space) ---
    v_codes, v_changed, v_total = _v_step(
        layer_no, layer, projection, source_dir, index,
        codes_np=seed_codes_np, effective_scales_np=base_scales_np, table=table,
        hessian_diag=hessian_diag, selected_experts=selected_experts,
    )
    arms["v_only"] = {
        "metric": metric_for(v_codes, base_scales_np),
        "changed_code_fraction": float(v_changed / v_total) if v_total else 0.0,
    }

    # --- Arm 4: PV alternating (K rounds of P then V) ---
    codes_np = seed_codes_np.copy()
    eff_scales_np = base_scales_np.copy()
    scale_delta_running = np.zeros_like(base_scales_np)
    round_trace = []
    for r in range(rounds):
        # P-step: fit scales for CURRENT codes.
        layer.codes = mx.array(codes_np)
        layer.set_continuous_sidecar(scale_delta=None, output_bias=None)
        scale_delta_np, p_res = _p_step(layer, x_mx, indices_mx, target_mx, ridge_strength=ridge_strength, max_abs_log_delta=max_abs_log_delta)
        scale_delta_running = scale_delta_np
        eff_scales_np = base_scales_np * np.exp(scale_delta_running)
        layer.set_continuous_sidecar(scale_delta=None, output_bias=None)
        p_metric = metric_for(codes_np, eff_scales_np)
        # V-step: reassign codes in the P-updated scaled space, then reload codes.
        codes_np, v_changed, v_total = _v_step(
            layer_no, layer, projection, source_dir, index,
            codes_np=codes_np, effective_scales_np=eff_scales_np, table=table,
            hessian_diag=hessian_diag, selected_experts=selected_experts,
        )
        v_metric = metric_for(codes_np, eff_scales_np)
        round_trace.append({
            "round": r,
            "after_p": p_metric,
            "after_v": v_metric,
            "p_improvement_ratio": float(p_res["improvement_ratio"]),
            "v_changed_code_fraction": float(v_changed / v_total) if v_total else 0.0,
        })
    # Final PV: one last P-step on the final codes so scales match final assignment.
    layer.codes = mx.array(codes_np)
    layer.set_continuous_sidecar(scale_delta=None, output_bias=None)
    scale_delta_np, p_res = _p_step(layer, x_mx, indices_mx, target_mx, ridge_strength=ridge_strength, max_abs_log_delta=max_abs_log_delta)
    eff_scales_np = base_scales_np * np.exp(scale_delta_np)
    layer.set_continuous_sidecar(scale_delta=None, output_bias=None)
    layer.codes = mx.array(seed_codes_np)  # restore seed on the resident layer
    arms["pv"] = {
        "metric": metric_for(codes_np, eff_scales_np),
        "rounds": rounds,
        "round_trace": round_trace,
    }

    return {"projection": projection, "selected_experts": selected_experts, "arms": arms}


def _verdict(projection_results: list[dict]) -> dict:
    def agg(arm):
        klds = [pr["arms"][arm]["metric"]["mean_kld"] for pr in projection_results]
        p999 = [pr["arms"][arm]["metric"]["p999_kld"] for pr in projection_results]
        rel = [pr["arms"][arm]["metric"]["rel_l2"] for pr in projection_results]
        return {"mean_kld": float(np.mean(klds)), "p999_kld": float(np.mean(p999)), "rel_l2": float(np.mean(rel))}

    rtn, p, v, pv = agg("rtn"), agg("p_only"), agg("v_only"), agg("pv")
    def pct(base, new):
        return float(100.0 * (base - new) / base) if base > 0 else 0.0
    beats_rtn = pv["mean_kld"] < rtn["mean_kld"]
    beats_p = pv["mean_kld"] < p["mean_kld"]
    beats_v = pv["mean_kld"] < v["mean_kld"]
    go = beats_rtn and beats_p and beats_v
    return {
        "aggregate": {"rtn": rtn, "p_only": p, "v_only": v, "pv": pv},
        "pv_vs_rtn_mean_kld_reduction_pct": pct(rtn["mean_kld"], pv["mean_kld"]),
        "pv_vs_p_only_mean_kld_reduction_pct": pct(p["mean_kld"], pv["mean_kld"]),
        "pv_vs_v_only_mean_kld_reduction_pct": pct(v["mean_kld"], pv["mean_kld"]),
        "pv_vs_rtn_p999_kld_reduction_pct": pct(rtn["p999_kld"], pv["p999_kld"]),
        "pv_beats_rtn": bool(beats_rtn),
        "pv_beats_p_only": bool(beats_p),
        "pv_beats_v_only": bool(beats_v),
        "decision": "GO" if go else "NO-GO",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Block-local PV-Tuning kill-gate spike for GLM-4.5-Air VQ-1.0.")
    parser.add_argument("--seed-artifact-dir", default="artifacts/glm-4.5-air-vq")
    parser.add_argument("--source-dir", default=None)
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--index-path", default=None)
    parser.add_argument("--model-id", default=GLM45_AIR_MODEL_ID)
    parser.add_argument("--layer", type=int, default=45)
    parser.add_argument("--projections", nargs="+", choices=GLU_PROJECTIONS, default=list(GLU_PROJECTIONS))
    parser.add_argument("--rounds", type=int, default=3, help="PV alternation rounds (P then V).")
    parser.add_argument("--ridge-strength", type=float, default=1.0e-3)
    parser.add_argument("--max-abs-log-delta", type=float, default=0.5)
    parser.add_argument("--max-tokens-per-prompt", type=int, default=48)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    source_dir = _resolve_source_dir(args.source_dir)
    config_path = Path(args.config_path) if args.config_path else source_dir / "config.json"
    index_path = Path(args.index_path) if args.index_path else source_dir / "model.safetensors.index.json"
    config = json.loads(config_path.read_text())
    index = load_safetensors_index(index_path)

    prompts = [
        "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr)//2]\n",
        "The mitochondria is the powerhouse of the cell because it",
        "import numpy as np\n\ndef softmax(x):\n    return np.exp(x) / np.sum(np.exp(x))\n\n# Compute attention scores",
        "In distributed systems, the CAP theorem states that a system cannot simultaneously guarantee",
        "class BinaryTree:\n    def __init__(self, value):\n        self.value = value\n        self.left = None\n",
    ]

    t0 = time.perf_counter()
    x_np = _capture_moe_input_states(
        source_dir=source_dir, config_path=config_path, index_path=index_path,
        layer=args.layer, prompts=prompts, max_tokens_per_prompt=args.max_tokens_per_prompt,
    )
    capture_seconds = time.perf_counter() - t0
    indices_np = _route_states(config, source_dir, index, args.layer, x_np, args.model_id)

    switch_glu = _load_switch_glu(Path(args.seed_artifact_dir), args.layer)

    projection_results = []
    for projection in args.projections:
        pr = _run_projection(
            layer_no=args.layer, projection=projection, switch_glu=switch_glu,
            source_dir=source_dir, index=index, x_np=x_np, indices_np=indices_np,
            rounds=args.rounds, ridge_strength=args.ridge_strength, max_abs_log_delta=args.max_abs_log_delta,
        )
        projection_results.append(pr)

    verdict = _verdict(projection_results)
    report = {
        "record_type": "pv_tuning_block_local_kill_gate",
        "model_id": args.model_id,
        "layer": args.layer,
        "projections": args.projections,
        "rounds": args.rounds,
        "ridge_strength": args.ridge_strength,
        "max_abs_log_delta": args.max_abs_log_delta,
        "token_count": int(x_np.shape[0]),
        "route_count": int(indices_np.size),
        "seed_artifact_dir": args.seed_artifact_dir,
        "capture_seconds": capture_seconds,
        "codebook": "frozen_e8_1bit_256entry",
        "note": "P-step tunes per-group scales only (frozen E8 codebook); V-step reassigns codes.",
        "projection_results": projection_results,
        "verdict": verdict,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out_json:
        Path(args.out_json).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
