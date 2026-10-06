# KEEP Re-Quantization Directions

Status: Phase 0 design note for the GLM-4.5-Air community-wow quality gap.

This note maps three zero-runtime quality levers onto KEEP's current code
surfaces. The accepted balanced RC already satisfies the size, speed, and
memory gates; the open gap is quality at the same routed 8-bit E8 VQ
representation used by the NAX fast path. The premise here is to re-quantize
the routed experts better, not to add decode-time work.

## Current KEEP Surfaces

KEEP's routed VQ materialization currently has four relevant layers:

- Source conversion: `src/mlx_vq/convert/stream_convert.py` reads each source
  expert projection, optionally applies deterministic RHT signs to the input
  dimension, and calls `quantize_weight_rtn(...)`.
- Quantizer: `src/mlx_vq/quant/rtn.py` computes per-row/per-group scales and
  nearest E8-family codewords. The promotable Air route uses `code_bits=8`;
  `code_bits=16` is implemented but rejected for routed runtime use in this
  quality lane.
- RHT candidate materialization: `src/mlx_vq/quality/rht_materialization.py`
  creates linked artifacts where selected routed projection groups are
  re-quantized after a deterministic random Hadamard transform.
- Calibration/imatrix path: `benchmarks/collect_glm45_air_imatrix.py` and
  `src/mlx_vq/quality/imatrix.py` collect routed activation squares per
  layer/projection/expert. `benchmarks/materialize_glm45_air_dynamic_imatrix_sweep.py`
  consumes the imatrix manifest for dynamic tiering plus bounded low-tier code
  reassignment through `imatrix_weighted_reassign_codes(...)`.

The declarative recipe hook for the legacy lineage is
`recipes/glm45air__dmx2p0__sc-l45__r26__legacy_lineage_20260701.yaml`:
`legacy_stream_convert_root -> legacy_collect_imatrix_01 ->
legacy_stream_convert_high_bit_01 -> legacy_materialize_sweep_01 -> sidecar
fit/eval gates`. New re-quantization levers should enter as new recipe ops or
as explicit extensions of these existing ops so `build_steps[]` lineage remains
auditable.

## A. Learned Rotation Instead Of Random RHT

Research basis: SpinQuant shows that random rotations have high quality
variance, while learned orthogonal rotations optimized on calibration loss can
beat random Hadamard initialization. Its `SpinQuant_no_had` variant uses only
mergeable rotations, so inference replaces original weights with rotated
quantized weights and does not require a forward-pass change.

KEEP mapping:

- Current random RHT is `rht_seed -> deterministic_rht_signs(...) ->
  apply_rht_np(weight, signs)` inside `convert_vq_group_from_safetensors(...)`.
- Replace the sign-only transform with a learned orthogonal matrix object:
  `rotation_kind=learned_orthogonal_v1`, `rotation_path`, `rotation_scope`,
  and `rotation_init_rht_seed`.
- For routed MLP gate/up projections, the practical first target is the input
  dimension that reads the residual stream. That matches the current RHT input
  transform and keeps stored weights shaped exactly as before.
- Down projection is trickier because its input is GLU output, not raw residual.
  A down-side learned transform should be a second phase unless paired gate/up
  and down transforms are proven equivalent for GLM's routed GLU. The safe first
  implementation should target `gate_proj` and `up_proj`.

Zero-runtime argument:

- The learned rotation is applied before E8 code assignment and fused into the
  stored codes/scales. Runtime still loads the same `codes`, `scales`, and
  `model.vq_codebook.e8` tensors through the same RAMP/NAX path.
- No routed decode kernel, code bit-width, group size, route sort, or sidecar
  operation changes.

Implementation sketch:

1. Add a small rotation artifact schema under a quality module, e.g.
   `src/mlx_vq/quality/learned_rotation.py`, that stores one orthogonal
   matrix per `(layer, projection)` target plus provenance.
2. Add a converter hook parallel to `rht_seed`, e.g. `rotation_matrix`, and
   apply `weight @ rotation_matrix` for row-vector semantics. This preserves
   the existing `out x in` source tensor shape.
3. Add a trainer/probe script that optimizes the rotation on a bounded
   calibration set. Start with source-oracle or quantized-proxy loss for
   selected layers; only escalate to end-to-end token loss once the local proxy
   path is stable.
4. Add manifest fields to both the group metadata and artifact
   `conversion-manifest.json`; do not mutate seed artifacts.

Verification plan:

- Unit: tiny sparse checkpoint proves learned-rotation materialization rewrites
  only target groups, links untouched groups, writes metadata, keeps loader
  compatibility, and differs from deterministic RHT.
- Numeric: rotation matrix is orthogonal within tolerance; output shapes and
  source-tensor read counts match current RHT materialization.
- Candidate gate: full report, selection, and holdout community-wow profile.
  Count a result only from a materialized candidate and repaired full-split
  evidence. If global top1 gains at least `+0.01` or route/math/code gains at
  least `+0.05`, compose it into the next candidate.

## B. Expert-Balanced Calibration

Research basis: MoEQuant's EBSS constructs calibration samples that balance
expert usage while staying near the model's own distribution. This addresses
under-calibrated rare experts, which matches KEEP's route-domain weakness.

KEEP mapping:

- Current fixed-prompt collection is `benchmarks/collect_glm45_air_imatrix.py`
  with prompt set `air_imatrix_calib_v1`.
- Current summaries already record selected expert counts and coverage
  fractions by layer, but prompt choice is static.
- EBSS should produce a new prompt set or prompt-selection manifest consumed by
  the same collector, rather than adding a new quantization path.

Cheapest first implementation:

1. Build `benchmarks/select_glm45_air_ebss_prompts.py` that scores candidate
   prompts by low perplexity proxy plus expert-balance improvement using
   existing route-summary records.
2. For the first promotable attempt, allow "expert-usage-rebalanced from
   existing prompt pools" rather than full self-sampling if self-sampling is
   too heavy. The manifest should state whether it is `ebss_self_sampled` or
   `ebss_rebalanced_existing_pool`.
3. Feed the selected prompt ids into `collect-imatrix` via existing
   `--prompt-id` / `--max-prompts` machinery, producing a new imatrix manifest.
4. Materialize through the same dynamic-imatrix sweep and normal eval gates.

Full EBSS follow-up:

- Add a self-sampling loop that keeps `w` branch candidates, caches cumulative
  log probability and expert-usage histograms, and scores candidates by the
  MoEQuant-style probability plus expert-balance term. Keep this data-only:
  no artifact mutation, no protected seed mutation.

Zero-runtime argument:

- EBSS changes only which calibration activations feed imatrix/rounding and
  tier allocation. The stored representation and RAMP decode path are unchanged.

Verification plan:

- Unit: prompt-selector scoring improves or preserves expert-usage standard
  deviation on synthetic route summaries while respecting prompt count and
  deterministic tie-breakers.
- Collection: imatrix manifest has the expected entry count
  `layers * projections * 128`, selected prompt ids, and lower or explicitly
  recorded expert-usage imbalance versus the fixed-prompt baseline.
- Candidate gate: full report, selection, holdout, and per-domain top1. Route
  top1 is the primary readout.

## C. Affinity-Guided Rounding

Research basis: MoEQuant's AGQ weights quantization error by token-expert
affinity. For a token routed to an expert with gate score `c_i`, the
quantization objective becomes affinity-weighted, and Hessian-style statistics
use activation rows scaled by affinity.

KEEP mapping:

- `build_activation_record(...)` already stores `router_scores` alongside
  `route_indices`.
- `accumulate_routed_projection_imatrix(...)` currently sums unweighted
  `x^2` for each routed expert and stores both mean and route-frequency-weighted
  vectors.
- `imatrix_weighted_reassign_codes(...)` already accepts a diagonal vector and
  calls the existing Hessian-weighted reassignment helper.

Required change:

- Add affinity-aware imatrix accumulation. For each flattened route row, use the
  corresponding router score as a non-negative weight:
  `importance_sum += c_i * x_i^2`.
- Store a separate tensor and manifest key such as
  `affinity_weighted_importance`; keep the existing unweighted tensors for
  backward compatibility and comparative evidence.
- For down projection, pair each per-route `down_input_rows[token, topk, :]`
  row with the same router score from `router_scores[token, topk]`.

Zero-runtime argument:

- AGQ only changes which existing 8-bit E8 codewords are selected and which
  low-tier experts are rerounded. Runtime tensor names, dimensions, codebook,
  and kernels do not change.

Verification plan:

- Unit: synthetic two-token/two-expert accumulation proves route scores change
  the diagonal vector, including the per-route down-input case.
- Materializer: dynamic-imatrix sweep can choose `importance_key` from
  `routing_weighted_importance` or `affinity_weighted_importance`, records it in
  summaries, and preserves existing behavior by default.
- Candidate gate: full split eval against baseline. Mean/p999 KLD and route
  top1 should be inspected together; a top1 drop below `0.5` or PPL ratio above
  `2x` is a bug signal, not a lever verdict.

## QTIP Boundary

QTIP is relevant as a future quality ceiling, but it is out of scope for this
session. It replaces the E8 VQ decode family with trellis-coded quantization and
a bitshift decoder. Even if the paper reports better 2-bit distortion than E8P,
that would require a new routed-critical-path representation and kernel. KEEP's
accepted balanced RC depends on the current 8-bit E8 code path and Lane S
envelope, so QTIP belongs in future runtime/kernel planning, not in this
zero-runtime quality pass.

## Recommended Execution Order

1. EBSS rebalanced calibration: lowest implementation risk and directly targets
   route-domain under-coverage.
2. AGQ diagonal rerounding: small extension of current imatrix/Hessian code and
   composable with EBSS.
3. Learned rotation: highest expected upside but needs a new orthogonal
   optimizer and more careful equivalence boundaries.

Each lever should be landed only with a materialized candidate, full repaired
report/selection/holdout evidence, per-domain top1 deltas, mean/p999 KLD, PPL
ratio, bpw, and a Lane S spot-check on the best composed candidate.
