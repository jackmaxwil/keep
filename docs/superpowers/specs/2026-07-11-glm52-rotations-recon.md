Read-only recon complete. No files were modified.

## Executive verdict

The generic VQ runtime already supports both dense orthogonal rotations and `rht_signs` end-to-end:

`transform source weights → quantize transformed weights → store transform tensor → load transform tensor → transform activations → run existing VQ/NAX kernel`.

GLM52 does not yet have a valid rotation-bearing recovery-candidate contract. The current recovery materializer writes exactly `{codes, scales, codebook}`, its resume checks require the seed and candidate tensor inventories to match, and the recovery auditor rejects every additional `rotation_matrix` or `rht_signs` tensor. Consequently, the frozen reevaluator would honor rotations if they reached the adapter, but today its prerequisite audit prevents such an artifact from reaching runtime.

Dense rotations do not disable NAX. They add a separate activation matmul immediately before NAX/gather VQMM. For a GLM52 gate or up projection, each distinct dense rotation costs:

- `6144 × 6144 = 37,748,736` multiply-accumulates per input token.
- Approximately `75.5 MFLOP/token` if multiply and add count separately.
- `150,994,944` bytes, about `144 MiB`, per FP32 matrix.
- Gate and up currently transform independently, so two separate matrices—or even one numerically shared matrix stored on both projection modules—cost about `151 MFLOP/token` unless the GLU path is taught to compute and reuse the shared transformed input.

The current RHT implementation cannot handle GLM52 gate/up width `6144`, because it explicitly requires a power-of-two dimension. It can handle the `2048`-wide down projection. Thus “cheap RHT for GLM52 gate/up” needs a new structured transform such as block/Kronecker `3 × 2048`, not merely generation of a 6144-element sign vector.

## 1. Existing runtime, storage, and fast-path support

### Runtime object and validation

`QuantizedVQSwitchLinear` accepts both fields at [switch_linear.py:100](/Users/jack.mazac/Developer/keep/src/mlx_vq/nn/switch_linear.py:100), specifically:

- `rht_signs` at [switch_linear.py:117](/Users/jack.mazac/Developer/keep/src/mlx_vq/nn/switch_linear.py:117)
- `rotation_matrix` at [switch_linear.py:118](/Users/jack.mazac/Developer/keep/src/mlx_vq/nn/switch_linear.py:118)

They are mutually exclusive at [switch_linear.py:162](/Users/jack.mazac/Developer/keep/src/mlx_vq/nn/switch_linear.py:162). RHT validation checks only the `(input_dims,)` shape there; the actual RHT operation later enforces power-of-two width. Dense rotation construction calls `validate_rotation_matrix_mx` at [switch_linear.py:166](/Users/jack.mazac/Developer/keep/src/mlx_vq/nn/switch_linear.py:166), then stores FP32 tensors at [switch_linear.py:205](/Users/jack.mazac/Developer/keep/src/mlx_vq/nn/switch_linear.py:205).

Important validation weakness: the MLX validator checks shape only and casts to FP32 at [rotation.py:25](/Users/jack.mazac/Developer/keep/src/mlx_vq/quant/rotation.py:25). It does not check finiteness or orthogonality. The NumPy validator does both at [rotation.py:7](/Users/jack.mazac/Developer/keep/src/mlx_vq/quant/rotation.py:7), including `RᵀR ≈ I` at [rotation.py:18](/Users/jack.mazac/Developer/keep/src/mlx_vq/quant/rotation.py:18).

### Forward ordering

The transform is applied to the activation before any VQ kernel selection:

- RHT: [switch_linear.py:312](/Users/jack.mazac/Developer/keep/src/mlx_vq/nn/switch_linear.py:312)
- Dense rotation: [switch_linear.py:315](/Users/jack.mazac/Developer/keep/src/mlx_vq/nn/switch_linear.py:315)
- Fast-path dispatch begins afterward: [switch_linear.py:318](/Users/jack.mazac/Developer/keep/src/mlx_vq/nn/switch_linear.py:318)

The dense operation is literally `x @ rotation_matrix` at [rotation.py:39](/Users/jack.mazac/Developer/keep/src/mlx_vq/quant/rotation.py:39).

The matching weight-side convention is implemented by `from_weights`:

- `W_rot = RHT(W)` at [switch_linear.py:507](/Users/jack.mazac/Developer/keep/src/mlx_vq/nn/switch_linear.py:507)
- `W_rot = W @ R` at [switch_linear.py:509](/Users/jack.mazac/Developer/keep/src/mlx_vq/nn/switch_linear.py:509)
- Quantization happens after transformation at [switch_linear.py:511](/Users/jack.mazac/Developer/keep/src/mlx_vq/nn/switch_linear.py:511)

For an orthogonal `R`, runtime computes `(xR)(WR)ᵀ = xRRᵀWᵀ = xWᵀ`, modulo quantization.

### Gather VQMM and NAX

Rotations do not cause a fallback from the ordinary gather path. The transformed `x` is passed to:

- M=1 E8 kernel at [switch_linear.py:329](/Users/jack.mazac/Developer/keep/src/mlx_vq/nn/switch_linear.py:329)
- General Metal gather VQMM at [switch_linear.py:349](/Users/jack.mazac/Developer/keep/src/mlx_vq/nn/switch_linear.py:349)
- Per-route M=1 kernel at [switch_linear.py:366](/Users/jack.mazac/Developer/keep/src/mlx_vq/nn/switch_linear.py:366)

The GLU sorted-prefill/NAX path also handles transforms explicitly. `_rht_input_for_projection` applies RHT or dense rotation at [glm4_moe_adapter.py:226](/Users/jack.mazac/Developer/keep/src/mlx_vq/models/glm4_moe_adapter.py:226). Gate and up invoke it separately at [glm4_moe_adapter.py:249](/Users/jack.mazac/Developer/keep/src/mlx_vq/models/glm4_moe_adapter.py:249) and [glm4_moe_adapter.py:268](/Users/jack.mazac/Developer/keep/src/mlx_vq/models/glm4_moe_adapter.py:268), after NAX/Metal implementation selection at [glm4_moe_adapter.py:194](/Users/jack.mazac/Developer/keep/src/mlx_vq/models/glm4_moe_adapter.py:194).

Therefore:

- `gather_vqmm`: retained.
- NAX E8/E8P sorted prefill: retained.
- Extra transform cost: unfused, paid before the selected kernel.
- Shared gate/up rotation reuse: missing.

### Storage and loading

Generic streaming conversion already transforms before quantization at [stream_convert.py:688](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/stream_convert.py:688), specifically [stream_convert.py:698](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/stream_convert.py:698)-[stream_convert.py:706](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/stream_convert.py:706).

It writes:

- `*.rht_signs` at [stream_convert.py:820](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/stream_convert.py:820)
- `*.rotation_matrix` at [stream_convert.py:822](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/stream_convert.py:822)
- Policy metadata at [stream_convert.py:839](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/stream_convert.py:839)-[stream_convert.py:849](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/stream_convert.py:849)

The loader retrieves both optional tensors at [load.py:235](/Users/jack.mazac/Developer/keep/src/mlx_vq/io/load.py:235), particularly [load.py:244](/Users/jack.mazac/Developer/keep/src/mlx_vq/io/load.py:244)-[load.py:258](/Users/jack.mazac/Developer/keep/src/mlx_vq/io/load.py:258).

`io/schema.py` does not model rotations as first-class tensor specifications. `VQTensorSpec` contains only dimensions, code bits, group size, and expert count at [schema.py:20](/Users/jack.mazac/Developer/keep/src/mlx_vq/io/schema.py:20). `QuantizationConfig.policy` is only a generic string dictionary at [schema.py:64](/Users/jack.mazac/Developer/keep/src/mlx_vq/io/schema.py:64). Storage works through conventional tensor names and policy strings, not an enforced rotation schema.

GLM52 uses this generic loader for all three routed projections at [glm52_vq_adapter.py:581](/Users/jack.mazac/Developer/keep/src/mlx_vq/models/glm52_vq_adapter.py:581)-[glm52_vq_adapter.py:597](/Users/jack.mazac/Developer/keep/src/mlx_vq/models/glm52_vq_adapter.py:597), including authenticated per-file paths at [glm52_vq_adapter.py:600](/Users/jack.mazac/Developer/keep/src/mlx_vq/models/glm52_vq_adapter.py:600)-[glm52_vq_adapter.py:628](/Users/jack.mazac/Developer/keep/src/mlx_vq/models/glm52_vq_adapter.py:628).

## 2. Existing generation and training code

### Random Hadamard

Deterministic SHAKE-256-derived signs exist at [rht.py:24](/Users/jack.mazac/Developer/keep/src/mlx_vq/quant/rht.py:24). NumPy and MLX transforms are at [rht.py:51](/Users/jack.mazac/Developer/keep/src/mlx_vq/quant/rht.py:51) and [rht.py:84](/Users/jack.mazac/Developer/keep/src/mlx_vq/quant/rht.py:84).

But `_validate_power_of_two_dim` rejects `6144` at [rht.py:10](/Users/jack.mazac/Developer/keep/src/mlx_vq/quant/rht.py:10). This blocks GLM52 gate/up transforms despite the environment probe listing 6144 among desired Hadamard widths elsewhere in the repository.

### Learned dense rotations

The generic trainer provides:

- Dense RHT initialization: [learned_rotation_training.py:35](/Users/jack.mazac/Developer/keep/src/mlx_vq/quality/learned_rotation_training.py:35)
- RTN reconstruction objective: [learned_rotation_training.py:40](/Users/jack.mazac/Developer/keep/src/mlx_vq/quality/learned_rotation_training.py:40)
- Cayley-style learned dense rotation: [learned_rotation_training.py:79](/Users/jack.mazac/Developer/keep/src/mlx_vq/quality/learned_rotation_training.py:79)
- Learned RHT sign-flip search: [learned_rotation_training.py:132](/Users/jack.mazac/Developer/keep/src/mlx_vq/quality/learned_rotation_training.py:132)

The dense objective refreshes RTN codes for `W @ R`, dequantizes them, rotates them back with `Rᵀ`, and minimizes unweighted mean squared reconstruction error at [learned_rotation_training.py:47](/Users/jack.mazac/Developer/keep/src/mlx_vq/quality/learned_rotation_training.py:47)-[learned_rotation_training.py:53](/Users/jack.mazac/Developer/keep/src/mlx_vq/quality/learned_rotation_training.py:53). Its update is based on the unweighted residual at [learned_rotation_training.py:109](/Users/jack.mazac/Developer/keep/src/mlx_vq/quality/learned_rotation_training.py:109)-[learned_rotation_training.py:118](/Users/jack.mazac/Developer/keep/src/mlx_vq/quality/learned_rotation_training.py:118).

That is not the GLM52 recovery objective. GLM52 recovery is explicitly `selection_diagonal_hessian_weighted_squared_error` at [glm52_recovery_materialize.py:774](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:774)-[glm52_recovery_materialize.py:789](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:789).

### GLM-4.5-Air integration

The existing CLI samples limited source rows and experts at [train_glm45_air_learned_rotation.py:67](/Users/jack.mazac/Developer/keep/benchmarks/train_glm45_air_learned_rotation.py:67)-[train_glm45_air_learned_rotation.py:102](/Users/jack.mazac/Developer/keep/benchmarks/train_glm45_air_learned_rotation.py:102).

It supports:

- Learned signs at [train_glm45_air_learned_rotation.py:136](/Users/jack.mazac/Developer/keep/benchmarks/train_glm45_air_learned_rotation.py:136)-[train_glm45_air_learned_rotation.py:183](/Users/jack.mazac/Developer/keep/benchmarks/train_glm45_air_learned_rotation.py:183)
- Dense Cayley rotations at [train_glm45_air_learned_rotation.py:185](/Users/jack.mazac/Developer/keep/benchmarks/train_glm45_air_learned_rotation.py:185)-[train_glm45_air_learned_rotation.py:230](/Users/jack.mazac/Developer/keep/benchmarks/train_glm45_air_learned_rotation.py:230)
- An Air-specific manifest at [train_glm45_air_learned_rotation.py:234](/Users/jack.mazac/Developer/keep/benchmarks/train_glm45_air_learned_rotation.py:234)

The Air materializer passes transforms into the generic converter at [learned_rotation_materialization.py:237](/Users/jack.mazac/Developer/keep/src/mlx_vq/quality/learned_rotation_materialization.py:237)-[learned_rotation_materialization.py:250](/Users/jack.mazac/Developer/keep/src/mlx_vq/quality/learned_rotation_materialization.py:250). Its manifest is explicitly Air-specific at [learned_rotation_materialization.py:294](/Users/jack.mazac/Developer/keep/src/mlx_vq/quality/learned_rotation_materialization.py:294).

This code is useful algorithmically but is not an authenticated GLM52 recovery lane.

## 3. GLM52 materializer insertion points and byte constraints

The core insertion point is `materialize_recovery_group` at [glm52_recovery_materialize.py:825](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:825).

Today it:

1. Requires the seed tensor inventory to be exactly `{codes, scales, codebook}` at [glm52_recovery_materialize.py:853](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:853)-[glm52_recovery_materialize.py:857](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:857).
2. Quantizes the unrotated source weight at [glm52_recovery_materialize.py:875](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:875)-[glm52_recovery_materialize.py:902](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:902).
3. Writes exactly the same three tensors at [glm52_recovery_materialize.py:921](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:921)-[glm52_recovery_materialize.py:925](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:925).

A rotation-aware implementation would transform `weights` before the expert quantization loop and add the transform tensor to the `save_file` mapping. The canonical name must use the same prefix returned by `_group_names`; that prefix is constructed at [glm52_recovery_materialize.py:765](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:765)-[glm52_recovery_materialize.py:771](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:771).

However, this is not a local one-line extension:

- Resume currently requires seed and target inventories to be identical at [glm52_recovery_materialize.py:1192](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:1192)-[glm52_recovery_materialize.py:1201](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:1201).
- Accounting counts every non-codebook tensor as “codes/scales” at [glm52_recovery_materialize.py:1477](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:1477)-[glm52_recovery_materialize.py:1491](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:1491).
- The full source-weight loop and materializer call are at [glm52_recovery_materialize.py:1857](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:1857)-[glm52_recovery_materialize.py:1886](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:1886).
- Logical payload enforcement is at [glm52_recovery_materialize.py:1908](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:1908)-[glm52_recovery_materialize.py:1924](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:1924).

The recovery auditor is an immediate blocker. It demands exactly `{codes, scales, codebook}` at [glm52_recovery_artifact.py:599](/Users/jack.mazac/Developer/keep/src/mlx_vq/validate/glm52_recovery_artifact.py:599)-[glm52_recovery_artifact.py:610](/Users/jack.mazac/Developer/keep/src/mlx_vq/validate/glm52_recovery_artifact.py:610). Any rotation tensor is reported as unexpected. Its payload total also excludes rotations by construction at [glm52_recovery_artifact.py:704](/Users/jack.mazac/Developer/keep/src/mlx_vq/validate/glm52_recovery_artifact.py:704)-[glm52_recovery_artifact.py:717](/Users/jack.mazac/Developer/keep/src/mlx_vq/validate/glm52_recovery_artifact.py:717).

“Seed byte compatibility” should therefore remain:

- Unselected groups: inherited byte-for-byte.
- Zero-importance experts in an E8 rewritten group: accepted seed codes/scales remain byte-identical.
- Selected rotation-bearing groups: intentionally become child-candidate files with a larger authenticated tensor inventory; they cannot satisfy inventory equality with the seed.
- Parent authority: accepted seed/composite hashes remain unchanged and separately pinned.
- Child identity: must cover transform kind, exact raw transform bytes, dtype, shape, SHA-256, transformed codes/scales, and the complete candidate manifest body.

## 4. Runtime cost at GLM52 shapes

Model dimensions are authoritative at [glm52-reap-504b-v2.yaml:5](/Users/jack.mazac/Developer/keep/models/glm52-reap-504b-v2.yaml:5)-[glm52-reap-504b-v2.yaml:11):

- Hidden width: 6144
- MoE intermediate width: 2048
- 75 sparse layers
- 168 experts, top-8 routing

Per input row:

| Projection | Rotation shape | MACs/token | Approx. FLOPs/token | FP32 payload |
|---|---:|---:|---:|---:|
| gate | 6144² | 37,748,736 | 75.5M | 144 MiB |
| up | 6144² | 37,748,736 | 75.5M | 144 MiB |
| down | 2048² | 4,194,304 | 8.39M | 16 MiB |

Key conclusions:

- Dense rotation preserves NAX but is not “free”; it is a dense, expert-independent matmul per projection input.
- Decode pays this once per projection per generated token, not once per selected expert.
- Prefill pays it over all flattened tokens before sorted-route NAX.
- Gate/up have the same source activation, but current code calls the transform separately. A genuinely shared rotation needs explicit reuse in `QuantizedVQSwitchGLU`.
- One shared gate/up matrix stored once is about 144 MiB. Separate gate/up matrices are about 288 MiB per layer.
- Across 75 layers, separate gate/up matrices alone are about 21.1 GiB; adding down matrices gives about 22.3 GiB. This is materially large even if it fits the nominal 112 GB whole-model cap.
- The dominant practical issue is runtime bandwidth/compute and duplicated gate/up work, not only artifact size.

RHT is asymptotically `O(d log d)` and stores only one sign per dimension, but the existing implementation cannot run at `d=6144`. For `d=2048`, current RHT is directly usable.

## 5. Frozen reevaluation behavior

The frozen recovery evaluator audits before importing or binding the runtime candidate at [run_glm52_recovery_wave1.py:821](/Users/jack.mazac/Developer/keep/benchmarks/run_glm52_recovery_wave1.py:821)-[run_glm52_recovery_wave1.py:872].

After audit, it loads the accepted composite and overlays recovered layers through the GLM52 adapter at [run_glm52_recovery_wave1.py:888](/Users/jack.mazac/Developer/keep/benchmarks/run_glm52_recovery_wave1.py:888)-[run_glm52_recovery_wave1.py:918]. That adapter invokes the generic loader, which automatically loads `rht_signs` and `rotation_matrix`.

Therefore:

- Runtime/evaluation semantics: yes, rotations are already honored.
- Current frozen flow in practice: no rotation-bearing recovered group can be evaluated, because the preceding audit rejects its extra tensor.
- No evaluator metric changes are needed. The work belongs in materialization, authentication/audit, candidate identity, payload accounting, and possibly shared-transform runtime optimization.

## (a) Exists versus missing

| Stage | Status | Evidence / gap |
|---|---|---|
| Generate deterministic RHT signs | Exists | [rht.py:24](/Users/jack.mazac/Developer/keep/src/mlx_vq/quant/rht.py:24), but only power-of-two dimensions |
| Learn RHT signs | Exists generically | [learned_rotation_training.py:132](/Users/jack.mazac/Developer/keep/src/mlx_vq/quality/learned_rotation_training.py:132), unweighted RTN objective |
| Learn dense orthogonal rotation | Exists generically | [learned_rotation_training.py:79](/Users/jack.mazac/Developer/keep/src/mlx_vq/quality/learned_rotation_training.py:79) |
| GLM52 selection/Hessian-weighted rotation learning | Missing | Existing learner has no importance/Hessian input |
| Quantize transformed source weights | Exists generically | [stream_convert.py:698](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/stream_convert.py:698) |
| Quantize transformed weights in GLM52 recovery | Missing | [glm52_recovery_materialize.py:875](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:875) uses untransformed weights |
| Store transform tensors | Exists generically | [stream_convert.py:820](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/stream_convert.py:820) |
| Store in GLM52 recovered groups | Missing | Fixed three-tensor write at [glm52_recovery_materialize.py:921](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:921) |
| Load transform tensors | Exists | [load.py:235](/Users/jack.mazac/Developer/keep/src/mlx_vq/io/load.py:235) |
| Apply before gather/NAX | Exists | [switch_linear.py:310](/Users/jack.mazac/Developer/keep/src/mlx_vq/nn/switch_linear.py:310), [glm4_moe_adapter.py:226](/Users/jack.mazac/Developer/keep/src/mlx_vq/models/glm4_moe_adapter.py:226) |
| Audit/authenticate GLM52 rotation tensors | Missing/blocking | [glm52_recovery_artifact.py:599](/Users/jack.mazac/Developer/keep/src/mlx_vq/validate/glm52_recovery_artifact.py:599) |
| Include transform bytes in candidate identity/accounting | Missing | Current accounting conflates all non-codebook tensors |
| Frozen candidate evaluation | Runtime-ready, audit-blocked | [run_glm52_recovery_wave1.py:832](/Users/jack.mazac/Developer/keep/benchmarks/run_glm52_recovery_wave1.py:832), then adapter bind at line 910 |
| Reuse shared gate/up dense transform | Missing | Gate/up transform calls are separate |

## (b) Concrete missing pieces and proposed signatures

### GLM52 training contract

```python
@dataclass(frozen=True)
class GLM52ProjectionRotation:
    layer: int
    projection: Literal["gate_proj", "up_proj", "down_proj"]
    transform_kind: Literal["dense_orthogonal", "rht", "block_rht"]
    rotation_id: str
    transform: np.ndarray
    transform_sha256: str
    parent_candidate_identity_sha256: str
    stats_manifest_sha256: str
    initial_weighted_loss: float
    final_weighted_loss: float
```

```python
def train_glm52_learned_rotation_np(
    weight: np.ndarray,
    importance: np.ndarray,
    *,
    group_size: int,
    code_bits: Literal[8, 16],
    steps: int,
    learning_rate: float,
    init_rotation: np.ndarray | None = None,
) -> LearnedRotationTrainResult:
    ...
```

The loss should be the same authenticated selection-diagonal-Hessian family as recovery, not the existing unweighted mean MSE.

For signs:

```python
def train_glm52_structured_signs_np(
    weight: np.ndarray,
    importance: np.ndarray,
    *,
    transform_spec: StructuredTransformSpec,
    group_size: int,
    code_bits: Literal[8, 16],
    steps: int,
    candidates_per_step: int,
    initial_signs: np.ndarray,
) -> LearnedRHTSignsTrainResult:
    ...
```

### Recovery materialization

```python
def materialize_recovery_group(
    *,
    source_weights: np.ndarray,
    importance: np.ndarray,
    seed_group_path: str | Path,
    output_path: str | Path,
    layer: int,
    projection: str,
    group_size: int = 512,
    code_bits: int = 8,
    rotation_matrix: np.ndarray | None = None,
    rht_signs: np.ndarray | None = None,
    rotation_authority: Mapping[str, object] | None = None,
    ...
) -> dict[str, object]:
    ...
```

Requirements:

- Enforce mutual exclusion.
- Validate dense matrices with `validate_rotation_matrix_np`.
- Transform source weights before importance-aware quantization.
- Write the transform tensor under the canonical projection prefix.
- Bind exact transform SHA-256 and raw byte length into provenance.
- Define whether importance is transformed into the rotated basis. A diagonal Hessian is not invariant under a dense rotation; simply quantizing `W @ R` with the original diagonal vector is mathematically inconsistent. Either use a full/structured Hessian surrogate or explicitly derive an approximated rotated diagonal.

### Manifest and audit

```python
@dataclass(frozen=True)
class RotationTensorRecord:
    kind: Literal["dense_orthogonal_v1", "hadamard_signs_v1", "block_hadamard_signs_v1"]
    tensor_name: str
    dtype: str
    shape: tuple[int, ...]
    tensor_payload_bytes: int
    sha256: str
    rotation_id: str
```

```python
def validate_rotation_tensor(
    handle: SafeTensorHandle,
    *,
    prefix: str,
    input_dim: int,
    expected: RotationTensorRecord | None,
) -> RotationTensorRecord | None:
    ...
```

Candidate identity and accounting should split:

```python
{
    "routed_codes_scales_bytes": ...,
    "routed_codebook_bytes": ...,
    "routed_rotation_bytes": ...,
}
```

Resume validation must compare against the child manifest’s expected tensor inventory, not require equality with the seed inventory.

### Shared gate/up runtime optimization

```python
def _shared_input_transform(
    self,
    x: mx.array,
) -> tuple[mx.array, mx.array]:
    """Return gate/up inputs, computing one transform when authorities match."""
```

A stronger representation would put one shared transform on `QuantizedVQSwitchGLU`, instead of duplicating it on both `gate_proj` and `up_proj`.

## (d) Random Hadamard versus learned dense rotation

Random/learned sign RHT and dense learned rotations share the generic storage/load/eval plumbing, but their GLM52 infrastructure needs differ materially.

### Dense learned rotation

Needs:

- GLM52-weighted training objective.
- Dense FP32/BF16 tensor authentication and substantial payload accounting.
- Dense activation matmul at inference.
- Preferably shared gate/up transform representation and computation.
- Orthogonality verification outside the MLX constructor.
- A decision about how the diagonal-Hessian objective transforms under `R`.

Does not need:

- A new VQ or NAX kernel. Existing kernels consume the already-rotated activation.

### Existing `rht_signs`

Needs:

- Tiny sign tensor and cheap transform.
- No dense matrix storage.
- Existing generic VQ/NAX integration already works for power-of-two widths.

But for GLM52:

- Down projection width `2048`: current implementation is usable.
- Gate/up width `6144`: current implementation is unusable because of [rht.py:10](/Users/jack.mazac/Developer/keep/src/mlx_vq/quant/rht.py:10).
- A GLM52 gate/up lane therefore needs a new transform definition and runtime, likely block/Kronecker Hadamard over the `6144 = 3 × 2048` structure.
- That transform kind must be represented distinctly in metadata and identity; calling it ordinary `hadamard_signs_v1` would misdescribe the operation.

Bottom line: the smallest infrastructure-complete first experiment is either a `2048`-wide down-projection RHT candidate, or one bounded gate/up dense rotation on a single layer with explicit shared-transform reuse. A “cheap 6144 RHT” is not currently available despite the presence of `rht_signs` plumbing.


Codex session ID: 019f547e-74c2-7e10-89e3-67643ab5f9a4
Resume in Codex: codex resume 019f547e-74c2-7e10-89e3-67643ab5f9a4
