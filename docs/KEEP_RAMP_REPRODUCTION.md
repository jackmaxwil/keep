# KEEP / RAMP Reproduction Draft

Status: DRAFT only. Not published. Not an upload instruction.

This document records the v0.1 reproduction path for the accepted balanced
GLM-4.5-Air KEEP artifact:

```text
artifacts/glm-4.5-air-dynamic3p0-r26-lora-l45-gud-math8-r4-init0p05-s12-lr0p5-w2-m1-nll0p5-20260701
```

KEEP is the compression method: KL-distilled Expert Encoding and Precision.
RAMP is the runtime: Routed Accelerated MoE Pipeline.

## Install

From the repository root:

```bash
uv sync --group dev
```

Build the native extension when using NAX-backed RAMP paths:

```bash
uv run cmake -S native/vq_nax_ext -B native/vq_nax_ext/build
uv run cmake --build native/vq_nax_ext/build -j
```

Run the cheap environment and artifact preflights:

```bash
scripts/glm45_air_rc.sh env-preflight
scripts/glm45_air_rc.sh preflight
```

The local GLM-4.5-Air source snapshot, accepted KEEP artifact, and generated
teacher/eval artifacts are not stored in git. If any of those inputs are absent,
the missing path is a local setup requirement, not a publication claim.

## RAMP Inference

Use RAMP for resident GLM-4.5-Air prefill/decode paths through this repository's
runtime and benchmark surfaces. The repo wrapper for the accepted KEEP artifact
is:

```bash
scripts/glm45_air_rc.sh benchmark-candidate --overwrite
```

This is a local runtime check, not publication evidence by itself. Any row with
pageouts or swapouts is invalid for acceptance evidence.

TODO: add the final public one-shot prompt-generation CLI once it is selected
for v0.1 publication.

## Rebuild From The r26 Seed

The exact accepted recipe command is:

```bash
keep build recipes/glm45air__dmx2p0__sc-l45__r26__20260701.yaml
```

If the shell has not activated the repo environment, run the same recipe through
`uv`:

```bash
uv run keep build recipes/glm45air__dmx2p0__sc-l45__r26__20260701.yaml
```

The recipe is the recorded reproduction path from the r26 seed to the accepted
balanced KEEP artifact. For graph validation or a no-model-work check:

```bash
uv run keep validate recipes/glm45air__dmx2p0__sc-l45__r26__20260701.yaml
uv run keep build recipes/glm45air__dmx2p0__sc-l45__r26__20260701.yaml --dry-run --no-hash
```

## Verification Gate Before Promotion

Do not promote or publish from this draft alone. Before any real promotion,
produce a clean full 128-row packet for report, selection, and holdout, then
rerun the runtime evidence required for Lane S.

The recorded accepted metrics for this draft are:

| Split | Clean rows | Mean KLD | Top1 | Mean PPL | Max PPL | p999 KLD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| report | 128/128 | 0.368578 | 0.795163 | 1.042025 | 1.741946 | 4.612547 |
| selection | 128/128 | 0.375400 | 0.806739 | 1.048529 | 1.803165 | 4.612547 |
| holdout | 128/128 | 0.358132 | 0.790883 | 1.025700 | 1.633145 | 4.612547 |

Lane S:

| Candidate median | Q2 quiet-control median | Ratio | Effective routed bpw | Dense routed experts | Unbound VQ experts | Non-expert dtype |
| ---: | ---: | ---: | ---: | --- | --- | --- |
| 1.5244784789829282s | 1.4189143960102228s | 1.0744x | 2.0359848484848486 | no | no | verified |

The honest limitation remains part of the publication packet: global top1
`0.795163` is below the internal community-wow target of `0.85`;
route/math/code domains are in the `0.40-0.50` range versus the `0.80` target;
and p999 KLD `4.612547` is above the internal `3.0` target. v0.1 is an honest
2-bit fast Mac artifact, not the wow candidate.
