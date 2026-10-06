Read-only recon complete. No files were modified.

The three additions can coexist cleanly if the materializer receives one immutable, authenticated `RecoveryRecipe`/policy object and applies the operations in this order:

```text
authenticate seed and tuning inputs
    → establish initial scales/codes
    → optional EBSS scale search
    → optional LDLQ code reassignment using the chosen scales
    → preserve zero-evidence experts where byte-compatible
    → publish groups
    → authenticate the complete manifest
```

The critical constraint is that the present GLM-5.2 statistics contain only per-coordinate diagonal importance. The existing LDLQ implementation requires a full positive-definite `H_in`. LDLQ therefore cannot truthfully be enabled merely by passing the existing diagonal vector: using `diag(importance)` produces no cross-codeword feedback and reduces the new lever to ordinary independent rounding.

## A. EBSS scale search

### Current scale path

The active function is [`quantize_weight_importance_aware()`](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:659).

Exact flow:

- Input/diagonal validation: [lines 676–683](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:676).
- Reshape into row/group/codeword/8D form: [lines 685–689](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:685).
- Select E8 or E8P table: [lines 694–698](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:694).
- Initial scale is exactly `max(abs(grouped)) / max(abs(table))`: [lines 699–701](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:699).
- Code assignment normalizes by the current scale and performs diagonal-Hessian nearest-code search:
  - E8: [lines 703–717](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:703).
  - E8P: [lines 719–730](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:719).
- Three alternating assignment/weighted least-squares scale updates by default: [lines 733–752](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:733).
- Final assignment and FP16 scale serialization: [lines 753–762](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:753).

The weighted least-squares update is:

```text
scale = Σ h_i w_i c_i / Σ h_i c_i²
```

over each row/group, implemented at [lines 735–752](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:735).

The generic RTN module has only two scale estimator names:

- `ScaleEstimator = Literal["max_abs", "percentile_99"]`: [`rtn.py:20`](/Users/jack.mazac/Developer/keep/src/mlx_vq/quant/rtn.py:20).
- Implementations: [`_estimate_group_scales()` lines 253–267](/Users/jack.mazac/Developer/keep/src/mlx_vq/quant/rtn.py:253).
- Public RTN parameter: [`quantize_weight_rtn(... scale_estimator="max_abs")`](/Users/jack.mazac/Developer/keep/src/mlx_vq/quant/rtn.py:270).

Do not add the proposed selection-driven search to `ScaleEstimator` in `rtn.py`: RTN is data-free, while this search consumes selection-derived Hessian weights. Calling it a generic RTN estimator would obscure the tuning boundary.

### Exact insertion seam

The search belongs inside `quantize_weight_importance_aware()`, immediately after the initial scales at [lines 699–701](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:699), using the existing assignment machinery at [lines 703–730](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:703).

For each candidate multiplier `m_j`:

```text
candidate_scale = base_scale * m_j
candidate_codes = assign(weight / candidate_scale)
error = Σ h_i * (w_i - candidate_scale * codeword_i)²
```

The sum reduces over the group’s codewords and eight coordinates. That is the exact diagonal-Hessian weighted objective already available in-loop; no forward pass or new tuning evidence is required.

Keep the candidate scale and codes for the lowest error per row/group. Tie-breaking must be deterministic—prefer the lowest candidate index—and non-finite candidates must fail closed.

Recommended signature:

```python
def quantize_weight_importance_aware(
    weight: np.ndarray,
    importance: np.ndarray,
    *,
    group_size: int = 512,
    code_bits: int = 8,
    iterations: int = 3,
    codeword_chunk_size: int = 8192,
    e8p_search_backend: str = "numpy",
    scale_search_multipliers: Sequence[float] | None = None,
) -> ImportanceAwareQuantizedWeight:
```

A private helper would keep allocation and objective logic isolated:

```python
def _search_group_scales_diagonal_hessian(
    *,
    grouped: np.ndarray,
    grouped_hessian: np.ndarray,
    initial_scales: np.ndarray,
    table: np.ndarray,
    code_bits: int,
    multipliers: Sequence[float],
    backend: str,
) -> tuple[np.ndarray, np.ndarray]:
```

Validation:

- Multipliers must be a nonempty, finite, strictly positive, duplicate-free tuple.
- Their exact ordered representation belongs in the authenticated policy/provenance.
- The search should include `1.0`, or reject the recipe, so it cannot silently omit the current scale.
- Decide explicitly whether the search replaces the three LS iterations or initializes them. For a stable first version, search once after max-abs initialization, then run the existing alternating LS loop. Record both `scale_search` and `scale_refinement_iterations`.
- `codeword_chunk_size` is currently accepted by the importance-aware function but not forwarded to either diagonal-Hessian search call; the new implementation should either wire it through or remove it separately rather than imply it controls EBSS memory.

Suggested authenticated declarations:

```json
{
  "scale_estimator": "selection_diagonal_hessian_ebss_v1",
  "scale_search_multipliers": "0.75,0.875,1.0,1.125,1.25",
  "scale_search_objective": "selection_diagonal_hessian_weighted_squared_error",
  "scale_refinement_iterations": "3"
}
```

Because policy values are currently required to be strings by [`_read_recovery_policy()`](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:1402), serialize the ordered multiplier tuple canonically rather than adding an unauthenticated list.

### Files/functions to touch

Core:

- [`glm52_recovery_materialize.py:659`](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:659): scale-search implementation and signature.
- [`_lever_provenance()` at line 774](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:774): declare scale-search identity and parameters.
- [`_recovery_quantization()` at line 797](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:797): put the same declarations into safetensors policy.
- [`materialize_recovery_group()` at line 825](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:825): forward the recipe to each expert.
- Resume checks at [`_validate_resumable_group()` lines 1218–1235](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:1218): require exact recipe/provenance equality.
- Manifest construction at [`build_recovery_conversion_manifest()` lines 1635–1666](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:1635).

Dependent authentication surface:

- The recovery auditor currently hard-codes the canonical estimator/objective at [`glm52_recovery_artifact.py:765–786`](/Users/jack.mazac/Developer/keep/src/mlx_vq/validate/glm52_recovery_artifact.py:765). It must accept an externally pinned recipe/lever allowlist, not silently accept arbitrary metadata.

## B. LDLQ-feedback lever

### Plumbing today

The materializer has one module-global lever:

- [`RECOVERY_LEVER = "selection_diagonal_hessian_importance_weighted_reround_v1"`](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:53).

It is copied into both authenticated surfaces:

- Safetensors `quantization_config.policy.recovery_lever` and `rounding_objective`: [lines 797–822](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:797).
- `glm52_recovery_provenance`: [lines 774–794](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:774).
- Per-group output record: [lines 910–964](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:910).
- Canonical group record reads the policy back out of the published safetensors file: [lines 1402–1469](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:1402).
- The conversion manifest requires identical policies across all selected groups and authenticates the body: [lines 1635–1666](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:1635).
- Resume rejects provenance/policy drift: [lines 1218–1235](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:1218) and [lines 1518–1579](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:1518).

The campaign declares levers as part of each experiment:

- Full75 current lever: [`recipe:64–73`](/Users/jack.mazac/Developer/keep/recipes/glm52_recovery_campaign_v1_20260711.yaml:64).
- Worst8 combined reround/rate levers: [lines 74–91](/Users/jack.mazac/Developer/keep/recipes/glm52_recovery_campaign_v1_20260711.yaml:74).
- Campaign schema admits exactly `recovery_levers` among experiment fields: [`config.py:61–72`](/Users/jack.mazac/Developer/keep/src/mlx_vq/recovery_campaign/config.py:61).

Audit/evaluation takes externally expected values rather than trusting the artifact:

- CLI audit arguments: [`run_glm52_recovery_wave1.py:1089–1105`](/Users/jack.mazac/Developer/keep/benchmarks/run_glm52_recovery_wave1.py:1089).
- Reevaluation arguments: [lines 1144–1150](/Users/jack.mazac/Developer/keep/benchmarks/run_glm52_recovery_wave1.py:1144).
- Auditor validates lever/policy: [`glm52_recovery_artifact.py:731–786`](/Users/jack.mazac/Developer/keep/src/mlx_vq/validate/glm52_recovery_artifact.py:731).
- It compares each replacement group’s exact provenance and policy at [`glm52_recovery_artifact.py:700–702`](/Users/jack.mazac/Developer/keep/src/mlx_vq/validate/glm52_recovery_artifact.py:700).

### LDLQ implementation and missing input

The proposed implementation already declares the exact lever:

- [`LEVER_NAME = "selection_hin_blockldlq_feedback_fp32_v1"`](/Users/jack.mazac/Developer/keep/src/mlx_vq/quality/ldlq_feedback.py:14).
- Public operation: [`ldlq_reassign_codes()` lines 170–180](/Users/jack.mazac/Developer/keep/src/mlx_vq/quality/ldlq_feedback.py:170).
- It requires `h_in.shape == (input_dim, input_dim)`: [lines 132–155](/Users/jack.mazac/Developer/keep/src/mlx_vq/quality/ldlq_feedback.py:132).
- It computes a full block-LDL feedback factor: [lines 42–89](/Users/jack.mazac/Developer/keep/src/mlx_vq/quality/ldlq_feedback.py:42).
- It starts from fixed scales, performs sequential feedback reassignment, and accepts LDLQ only when full `H_in` error does not worsen: [lines 193–233](/Users/jack.mazac/Developer/keep/src/mlx_vq/quality/ldlq_feedback.py:193).
- Optional block refinement and final stats are at [lines 235–274](/Users/jack.mazac/Developer/keep/src/mlx_vq/quality/ldlq_feedback.py:235).

By contrast, current recovery stats reach the materializer as `importance[expert, in]`: [`materialize_recovery_group()` lines 844–851](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:844). The expert loop passes only that vector: [lines 875–889](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:875).

Therefore the LDLQ path needs a separately authenticated full-Hessian authority. It must not synthesize `np.diag(importance)` and claim the LDLQ lever.

Recommended signatures:

```python
RoundingLever = Literal[
    "selection_diagonal_hessian_importance_weighted_reround_v1",
    "selection_hin_blockldlq_feedback_fp32_v1",
]

def materialize_recovery_group(
    *,
    source_weights: np.ndarray,
    importance: np.ndarray,
    h_in: np.ndarray | None = None,  # [experts, in, in] or explicitly shared [in, in]
    rounding_lever: str = DEFAULT_RECOVERY_LEVER,
    ldlq_sweeps: int = 0,
    ...
) -> dict[str, object]:
```

Per expert, at the existing branch [lines 877–900](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:877):

1. Run importance-aware scale/code fitting, including optional EBSS.
2. If `rounding_lever == LDLQ_LEVER`, require an authenticated full `H_in` for that expert.
3. Call `ldlq_reassign_codes(source_weight, fitted_scales, h_in, codebook=full_table, ...)`.
4. Keep the fitted scales; replace only the codes.
5. Preserve the LDLQ implementation’s no-worse acceptance result.
6. Record aggregate LDLQ statistics in the group record or a separately authenticated per-expert record. Do not put floating statistics only in logs.

The rounding objective for the LDLQ recipe should change to something like:

```text
selection_hin_weighted_squared_error_with_blockldlq_feedback
```

The old diagonal objective must not remain in the policy for an LDLQ result.

Authentication implications:

- Add `rounding_lever` and its exact parameters to `_lever_provenance()` and `_recovery_quantization()` rather than reading the module constant.
- Add the full-Hessian manifest SHA-256, entry SHA-256, method, split evidence, damping/regularization, and shared-vs-per-expert contract to provenance.
- Extend the stats loader or add a distinct `_load_authenticated_hin_stats()`; do not overload diagonal `STATS_ENTRY_KEYS` at [`glm52_recovery_materialize.py:118–133`](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:118).
- Resume identity must include the LDLQ authority digest and `ldlq_sweeps`.
- The auditor’s current single allowed lever at [`glm52_recovery_artifact.py:765–786`](/Users/jack.mazac/Developer/keep/src/mlx_vq/validate/glm52_recovery_artifact.py:765) must become an exact recipe validator with a closed allowlist.
- The campaign experiment needs the LDLQ lever in `recovery_levers`, a named full-Hessian tuning authority, and explicit argv containing its expected digest.

## C. Prior recovery artifact as seed

### Baseline seed contract today

`_validate_seed_manifest()` accepts exactly one old materialization schema:

- Exact top-level key equality: [`glm52_recovery_materialize.py:541–547`](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:541).
- Required header:
  - `schema_version: 1`
  - `record_type: glm52_modelopt_nvfp4_materialization_manifest`
  - pinned profile/model/revision/config/index
  - `planned_vq_groups: 225`
  - `ready_vq_groups: 225`

  at [lines 548–561](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:548).
- Exactly 225 canonical layer 3–77 × three projections: [lines 529–538](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:529) and [lines 563–565](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:563).
- Exact group-record keys are declared at [lines 145–150](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:145).
- Canonical order, filename, positive byte size, and SHA-256 are checked at [lines 567–581](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:567).
- Each selected seed group is stable-opened, hash/size checked, and snapshotted before model work: [lines 585–619](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:585).

The top-level run further requires that the caller’s expected seed digest equal the repository-pinned baseline digest: [lines 1693–1701](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:1693), then stable-reads `seed_root/conversion-manifest.json` and compares it before validation: [lines 1724–1746](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:1724).

### Full75 recovery artifact today

The live Full75 artifact is now complete:

- Conversion manifest:
  [`artifacts/quality/glm52-recovery-wave1-mixed75-artifact-20260710/conversion-manifest.json`](/Users/jack.mazac/Developer/keep/artifacts/quality/glm52-recovery-wave1-mixed75-artifact-20260710/conversion-manifest.json)
- Audit:
  [`glm52-recovery-wave1-mixed75-artifact-audit-20260710.json`](/Users/jack.mazac/Developer/keep/artifacts/quality/glm52-recovery-wave1-mixed75-artifact-audit-20260710.json)

Observed contract:

- `schema_version: 2`
- `record_type: glm52_recovery_conversion_manifest`
- 225 group records
- all 225 are replacements
- `mixed_artifact.replacement_count: 225`
- `mixed_artifact.inherited_group_count: 0`
- raw conversion-manifest SHA-256:
  `544163a74793f78f739649a5a23328ea0d161ed4d667309f99623987b01ba082`
- audit SHA-256:
  `c65f98ed2328cbd765d5cee46c6709ae72fa6d716186335f336ea24c0ae9e638`
- audit says `audit_pass: true`, `group_count: 225`, and candidate identity
  `60e5495aa305efbb0ea38db59d7d1a51060d906eddb158501a8c5a40b448e3f6`.

Layout:

```text
glm52-recovery-wave1-mixed75-artifact-20260710/
├── conversion-manifest.json
├── recovered-groups/
│   └── 225 regular safetensors files
└── artifact/
    └── 225 relative symlinks to ../recovered-groups/...
```

The existing artifact validator enforces those links as relative symlinks and resolves them only into authenticated seed/recovered roots at [`glm52_recovery_artifact.py:411–456`](/Users/jack.mazac/Developer/keep/src/mlx_vq/validate/glm52_recovery_artifact.py:411).

The recovery manifest’s exact schema and authenticated `manifest_body_sha256` checks are at [`glm52_recovery_artifact.py:796–831`](/Users/jack.mazac/Developer/keep/src/mlx_vq/validate/glm52_recovery_artifact.py:796). Its `mixed_artifact` roots/counts and underlying baseline seed digest are validated at [lines 860–903](/Users/jack.mazac/Developer/keep/src/mlx_vq/validate/glm52_recovery_artifact.py:860).

### Minimal fail-closed seed extension

Use a mutually exclusive seed mode, not polymorphic interpretation of `--seed-artifact-dir`:

```text
--seed-artifact-dir BASELINE_DIR
--expected-seed-manifest-sha256 BASELINE_SHA

or

--seed-recovery-artifact RECOVERY_DIR
--expected-seed-recovery-manifest-sha256 RAW_RECOVERY_MANIFEST_SHA
--seed-recovery-audit-json AUDIT_JSON
--expected-seed-recovery-audit-sha256 AUDIT_SHA
--expected-seed-candidate-identity-sha256 CANDIDATE_ID
```

The direct materializer parser seam is [`glm52_recovery_materialize.py:1984–2000`](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:1984). The campaign wrapper currently requires only `--seed-artifact-dir` and baseline digest at [`run_glm52_recovery_wave1.py:1108–1119`](/Users/jack.mazac/Developer/keep/benchmarks/run_glm52_recovery_wave1.py:1108), while the campaign command allowlist hard-requires those baseline options at [`config.py:91–115`](/Users/jack.mazac/Developer/keep/src/mlx_vq/recovery_campaign/config.py:91). Both need mutually exclusive, complete option bundles.

Proposed internal type:

```python
@dataclass(frozen=True)
class AuthenticatedSeed:
    kind: Literal["accepted_baseline", "audited_recovery"]
    root: Path
    artifact_root: Path
    records: Mapping[tuple[int, str], SeedGroupRecord]
    manifest_sha256: str
    candidate_identity_sha256: str | None
    audit_sha256: str | None
    source_lineage: Mapping[str, str]
    layer_code_bits: Mapping[int, int]
```

Proposed validator split:

```python
def _authenticate_baseline_seed(...) -> AuthenticatedSeed
def _authenticate_recovery_seed(...) -> AuthenticatedSeed
```

For a recovery seed, fail closed unless all of the following hold:

- Stable-read raw conversion manifest equals the externally supplied digest.
- Manifest body hash is valid.
- Exact record type/status/resumable/schema are valid.
- Recovery audit JSON stable-read equals its externally supplied digest.
- Audit has `audit_pass is True`.
- Audit’s `raw_recovery_manifest_sha256` equals the raw manifest digest.
- Audit’s candidate identity equals the externally pinned candidate identity.
- `verify_current_identity()` succeeds immediately before snapshotting and again before publication.
- Audit proves exactly 225 canonical groups.
- Every group’s resolved file hash/size matches the audit record.
- Source lineage, source inventories, selection-only stats authority, accepted baseline composite identity, and accepted composite audit digest remain compatible with the requested child run.

Do not feed `artifact/*.safetensors` symlinks directly into `_snapshot_seed_groups()`: its `AuthenticatedFile.open()` path at [lines 598–604](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:598) is designed for authenticated regular files. Resolve each link through the audited classification logic, then snapshot the verified regular target under `recovered-groups/` or the recovery seed’s own authenticated inherited root.

### Byte compatibility and inherited links

Current selected-group seed checks are baseline-specific:

- exact tensor names: [lines 853–860](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:853);
- seed codes must be `uint8`, scales FP16, and codebook canonical E8: [lines 866–873](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:866).

That works for Full75 because it is all E8. It does not work generically for a prior mixed E8/E8P recovery seed. Replace it with a per-group seed contract derived from the authenticated recovery audit:

```python
def _validate_seed_group_tensor_contract(
    *,
    seed_path: Path,
    expected_shape: tuple[int, int, int],
    seed_code_bits: Literal[8, 16],
) -> SeedTensorContract
```

Rules:

- Seed `uint8` ↔ canonical E8; `uint16` ↔ canonical E8P.
- Scale shape/dtype must remain canonical.
- For zero-importance experts, preserve seed bytes only when `seed_code_bits == output_code_bits`.
- On an explicit rate transition, zero-evidence experts cannot be copied across dtypes. Preserve the current policy’s source-RTN fallback for E8→E8P, or reject the transition if the recipe forbids data-free fallback.
- Record that choice in `zero_importance_policy`; resume currently hard-codes only `"preserve_seed_expert_bytes"` or `"source_rtn_e8p"` at [lines 1596–1607](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:1596).

For the child mixed tree, `build_mixed_artifact_tree()` should receive `AuthenticatedSeed.artifact_root`, not the recovery wrapper directory. Its result must nevertheless preserve both parent identities:

```json
{
  "seed_kind": "audited_recovery",
  "seed_recovery_manifest_sha256": "...",
  "seed_recovery_audit_sha256": "...",
  "seed_candidate_identity_sha256": "...",
  "accepted_baseline_seed_manifest_sha256": "ba1d...",
  "parent_artifact_root": ".../parent/artifact"
}
```

The current manifest adds only the original seed digest at [`glm52_recovery_materialize.py:1902–1907`](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py:1902); this must become the typed seed descriptor above. Inherited child links should point to the parent recovery artifact’s authenticated `artifact/<filename>` only if the validator supports a safe link-to-link chain. The simpler and safer design is to resolve the parent link during authentication and create the child link directly to its verified regular target. Then the child auditor needs an explicit allowlisted collection of authenticated inherited roots instead of assuming one baseline root.

## Campaign matrix implications

The present matrix declares EBSS but does not have an executable transition:

- EBSS depends on Worst8 at [`recipe:92–109`](/Users/jack.mazac/Developer/keep/recipes/glm52_recovery_campaign_v1_20260711.yaml:92).
- Its `transition` is currently `null`: [`recipe:109`](/Users/jack.mazac/Developer/keep/recipes/glm52_recovery_campaign_v1_20260711.yaml:109).
- Worst8’s current transition still seeds directly from the accepted baseline, not Full75: [`recipe:170–208`](/Users/jack.mazac/Developer/keep/recipes/glm52_recovery_campaign_v1_20260711.yaml:170). Thus `depends_on: [full75-e8]` is orchestration ordering, not actual parent-artifact lineage.

A combined-seed or EBSS transition needs all of these declared in the matrix and included in its semantic fingerprint:

- Parent experiment name.
- Parent raw recovery-manifest path and SHA-256.
- Parent audit path and SHA-256.
- Parent candidate identity.
- Expected parent rate policy.
- Exact selected groups.
- Exact ordered recovery levers.
- Exact scale-search multiplier tuple and iteration count.
- Full-Hessian authority and digest for LDLQ, if enabled.
- Expected child rate policy and payload bytes.
- A non-null transition whose argv contains the complete mutually exclusive recovery-seed authority bundle.

The config parser is exact-key/fail-closed at [`config.py:156–166`](/Users/jack.mazac/Developer/keep/src/mlx_vq/recovery_campaign/config.py:156), so new typed fields must be deliberately added to `_EXPERIMENT_FIELDS` and `_TRANSITION_FIELDS`; otherwise the recipe correctly rejects them. Conversely, putting these only in free-form argv would authenticate the command bytes but leave the semantic experiment matrix unable to prove parent/recipe consistency.

## Conflict-free landing sequence

All core behavior can land in the one requested file, [`glm52_recovery_materialize.py`](/Users/jack.mazac/Developer/keep/src/mlx_vq/convert/glm52_recovery_materialize.py), if implemented in this order:

1. **Recipe/provenance refactor first.**
   Replace the module-global semantic assumption with an immutable recipe passed through `quantize_weight_importance_aware()`, `materialize_recovery_group()`, resume validation, canonical records, and manifest construction. Keep the current recipe as the exact default so existing artifacts remain reproducible.

2. **Recovery seed second.**
   Introduce `AuthenticatedSeed`, separate baseline/recovery authentication, per-group seed code-bit contracts, and parent identity fields. This establishes the correct initial bytes/scales for both later algorithms.

3. **EBSS third.**
   Search candidate scales from the authenticated parent state, using the existing diagonal objective. Authenticate candidate multipliers, tie policy, and refinement count.

4. **LDLQ fourth.**
   Apply LDLQ only after the final EBSS scales are chosen. Require an authenticated full `H_in`; replace codes only; record full-Hessian authority and LDLQ statistics. Refuse the lever when only diagonal importance exists.

5. **Resume and manifest last within the same change.**
   Make recipe identity, parent seed identity, scale search, full-Hessian identity, and LDLQ parameters part of exact resume equality. A run with any changed component must not reuse old recovered groups.

6. **Then update dependent surfaces atomically.**
   The campaign wrapper/config, recovery auditor, recipe, expected-policy JSON, and tests must recognize the new closed set. Publishing materializer support without corresponding auditor support would create artifacts that cannot pass the existing authenticated audit—which is a safe failure, but not an integrated transition.

The main conflict to avoid is letting each addition independently redefine `RECOVERY_LEVER`, `scale_estimator`, or seed identity. One typed recipe and one typed seed descriptor should be the sole source for safetensors metadata, group records, resume checks, conversion manifest, audit expectations, and campaign declarations.


Codex session ID: 019f547e-7452-7d70-83fd-e4d91d47b638
Resume in Codex: codex resume 019f547e-7452-7d70-83fd-e4d91d47b638
