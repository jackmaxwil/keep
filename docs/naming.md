# Project identity & naming convention

## Method / project: KEEP
**KEEP** = **K**L-distilled **E**xpert **E**ncoding & **P**recision.

Grounded, REAP-style (REAP = Router-weighted Expert Activation Pruning):
- **K — KL-distilled:** recovery via continuous sidecars / low-rank adapters / router-KD trained to KL-match the teacher; the acceptance metric is mean KLD.
- **E — Expert:** operates on the routed experts (the parameter bulk).
- **E — Encoding:** E8P vector-quantization / codebook encoding of the weights.
- **P — Precision:** dynamic, importance-matrix–allocated mixed precision (bit-budget tiering).

One-line method: *importance-matrix–allocated mixed-precision VQ on the routed experts, recovered by KL-distillation to the teacher, on MLX.*

Pairing logic: **REAP** cuts experts down; **KEEP** preserves the fidelity of what remains. Cut, then keep. (Both share "Expert" as the scope word.)

## Engine: RAMP
**RAMP** = **R**outed **A**ccelerated **M**oE **P**ipeline — the MLX/Metal runtime that serves low-bit routed MoE fast and memory-clean.
- **R — Routed:** top-k token->expert routing, no dense expert materialization.
- **A — Accelerated:** NAX/Metal-accelerated kernels (prefill + decode); the Kernel-Parity / NAX / Lane S work.
- **M — MoE:** Mixture-of-Experts serving.
- **P — Pipeline:** the end-to-end runtime/serving pipeline.

RAMP is **compression-agnostic** — it runs any low-bit routed MoE; KEEP is one producer that feeds it.

## Two pillars / packages
- **KEEP** (method): produces compressed artifacts; tags published models. Package `keep` (subpackages `keep.vq`, `keep.quant`, `keep.quality`, ...).
- **RAMP** (engine): serves them. Package `ramp` (kernels, routing, prefill/decode, Lane S).

The repo is `keep`, containing both `keep` and the reusable `ramp` package.
`mlx_vq` is the original package. `keep` and `ramp` re-export from it while code moves, and new code goes in `keep` or `ramp`.

## Published model names
Format: `<Base>[-REAP]-KEEP-<size>[-<bpw>]`
- `GLM-4.5-Air-KEEP` (KEEP only, no prune)
- `DeepSeek-V4-Flash-KEEP-2.03bpw`

The `-KEEP-` tag marks our method on shared/published artifacts (the way `-REAP-` marks Cerebras' prune).

## Internal artifact / run naming
KEEP is implied internally (the whole repo is KEEP), so internal run dirs use a structured, sortable, **immutable** scheme:

```
<base>__<repr><bpw>__<recovery>__<rev>__<date>
```

| Field | Meaning | Examples |
|---|---|---|
| `base` | base model + prune state | `glm45air`, `dsv4flash`, `glm53flash` |
| `repr`+`bpw` | representation + effective bits/weight (`p` = decimal point) | `dmx2p4` (dynamic mixed precision @2.4), `e8p2p0`, `q8` |
| `recovery` | recovery method(s), `+`-joined, `none` if raw | `none`, `rkd`, `sc-l45`, `lora16`, `sc+lora16`, `yaqa` |
| `rev` | short monotonic revision | `r26`, `r01` |
| `date` | YYYYMMDD | `20260701` |

Rules: lowercase, no spaces; `p` for decimals; `+` only inside `recovery`; **names immutable** — never encode acceptance/clean status in the name. `bpw` = routed effective bpw (headline budget; whole-model bpw in the manifest).

Examples:
- r26 (accepted Air): `glm45air__dmx2p0__sc-l45__r26__20260701`  ->  publishes as `GLM-4.5-Air-KEEP`

## Canonical manifest tags (mutable; not in the name)
```json
{
  "method": "KEEP",
  "base_model": "glm-4.5-air",
  "prune": {"method": "none|reap", "keep": null, "of": null},
  "repr": "dynamic_mixed_precision|e8p_vq|scalar",
  "bpw_effective_routed": 2.036,
  "bpw_effective_whole": 2.9,
  "recovery": ["sidecar:layer45", "router_kd", "lora:r16"],
  "status": "seed|candidate|accepted|rejected|superseded",
  "eval": {"report": "clean|dirty|pending", "selection": "...", "holdout": "..."},
  "lane_s": {"ratio_q2": 1.146, "pass": true},
  "rev": "r26", "date": "20260701",
  "seed_artifact": "<path>", "seed_immutable_proven": true
}
```
`status` and `eval` are the mutable fields that change as a candidate is evaluated/promoted; the directory name never changes.
