# KEEP DeepSeek-V4-Flash Cutover Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decouple KEEP's model-definition layer from GLM-5.2 and onboard DeepSeek-V4-Flash-0731 as the first family registered through generalized, registry-based surfaces.

**Architecture:** Three closed, hand-edited coupling points become open registries: the converter-kind whitelist in `profiles.py`, the per-family symbol soup in `models/__init__.py`, and the absent FP8 source path. New family code lives physically under `src/ramp/models/` and `src/keep/convert/` (the documented cutover direction — `mlx_vq` becomes the compatibility shim for new code, per `docs/KEEP_NEW_MODEL_FAMILY_TEMPLATE.md`). The full DeepSeek VQ adapter (model class, binders, teacher) is a follow-on plan once weights are downloaded and measured; this plan builds everything that does not require the weights.

**Tech Stack:** Python 3.12 via uv, pytest, MLX/mlx-lm 0.31.3, numpy, PyYAML, CloudFormation.

## Global Constraints

- Tests run as: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest <path> -q`
- Commit with explicit paths, never `git add .`
- Commits end with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`
- No push, merge, publish, or Hugging Face upload without explicit approval
- `runs/` and root `glm52-*.md` handoff files are protected — do not delete or overwrite
- Existing `mlx_vq.*` imports must keep working (compatibility shim direction: new code in `ramp`/`keep`, old names re-exported from `mlx_vq`)
- Target model identity: `deepseek-ai/DeepSeek-V4-Flash-0731`, `model_type deepseek_v4`, 43 layers, hidden 4096, vocab 129280, 256 routed experts + 1 shared, top-6, `moe_intermediate_size` 2048, FP8 e4m3 weights in 128x128 blocks with ue8m0 scales, `num_nextn_predict_layers` 1

---

### Task 1: Commit the pending SkyPilot submission fixes

The working tree already contains a tested bug fix (`--async` → `--detach-run` for `sky jobs launch` under pinned SkyPilot 0.13.0) plus a launch-capability waiver flag, and an untracked cache-seed task file. All 313 tests in the integration module pass. Land them before any refactor so the fix is not entangled with cutover changes.

**Files:**
- Modify: none (already modified: `aws/glm52-gpu/scripts/submit_sky_campaign.py`, `tests/test_glm52_sky_submission_integration.py`)
- Add: `aws/glm52-gpu/skypilot/glm52-cache-seed.yaml`

**Interfaces:**
- Consumes: nothing from other tasks
- Produces: clean working tree for subsequent tasks

- [ ] **Step 1: Run the integration tests**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_glm52_sky_submission_integration.py -q`
Expected: `313 passed`

- [ ] **Step 2: Commit**

```bash
git add aws/glm52-gpu/scripts/submit_sky_campaign.py tests/test_glm52_sky_submission_integration.py aws/glm52-gpu/skypilot/glm52-cache-seed.yaml
git commit -m "fix(glm52): detach managed launch and add capability waiver

sky 0.13.0 'jobs launch' takes --detach-run, not --async. Add
--waive-cache-seed-launch-capability to skip only the deployed
capability proof while keeping the one-shot launch claim.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Commit the pivot documentation

**Files:**
- Add: `keep-deepseek-v4-flash-pivot-20260811.md` (already written at repo root)
- Add: `docs/superpowers/plans/2026-08-11-keep-deepseek-v4-flash-cutover.md` (this plan)

**Interfaces:**
- Consumes: nothing
- Produces: the decision record every later task cites

- [ ] **Step 1: Commit**

```bash
git add keep-deepseek-v4-flash-pivot-20260811.md docs/superpowers/plans/2026-08-11-keep-deepseek-v4-flash-cutover.md
git commit -m "docs: record GLM-5.2 to DeepSeek-V4-Flash pivot

GLM-5.2 retired: run dead on p5 InsufficientInstanceCapacity, weights
purged from S3 (1.3 TB), watchdog disabled, controller terminated.
DeepSeek-V4-Flash-0731 is the new primary target (AA index 52 at 284B
total / 13B active vs GLM-5.2's 53 at 753B/40B).

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Open the converter registry in profiles.py

`ConverterKind` is a closed `Literal` plus `ALLOWED_CONVERTERS` frozenset (`src/mlx_vq/models/profiles.py:14-16`). A new family cannot exist without editing the type. Replace with a mutable registry seeded with the existing three kinds, and register the DeepSeek kind.

**Files:**
- Modify: `src/mlx_vq/models/profiles.py:14-16` and the `__post_init__` converter check
- Test: `tests/test_model_profiles.py`

**Interfaces:**
- Consumes: nothing
- Produces: `register_converter(kind: str) -> None`, `converter_kinds() -> frozenset[str]` in `mlx_vq.models.profiles`. Converter `"deepseek_v4_vq_groups"` registered by default. Task 5's profile YAML uses that kind.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_model_profiles.py`)

```python
def test_converter_registry_is_open():
    from mlx_vq.models import profiles

    assert "glm52_vq_groups" in profiles.converter_kinds()
    assert "deepseek_v4_vq_groups" in profiles.converter_kinds()

    profiles.register_converter("test_family_vq_groups")
    try:
        assert "test_family_vq_groups" in profiles.converter_kinds()
    finally:
        profiles.unregister_converter("test_family_vq_groups")
    assert "test_family_vq_groups" not in profiles.converter_kinds()


def test_register_converter_rejects_blank():
    import pytest
    from mlx_vq.models import profiles

    with pytest.raises(profiles.ProfileError):
        profiles.register_converter("")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_model_profiles.py -q -k converter`
Expected: FAIL with `AttributeError: ... has no attribute 'converter_kinds'`

- [ ] **Step 3: Implement**

In `src/mlx_vq/models/profiles.py`, replace lines 14-16:

```python
ConverterKind = str

_CONVERTER_KINDS: set[str] = {
    "glm_stream",
    "qwen_moe_groups",
    "glm52_vq_groups",
    "deepseek_v4_vq_groups",
}


def register_converter(kind: str) -> None:
    if not isinstance(kind, str) or not kind:
        raise ProfileError("converter kind must be a non-empty string")
    _CONVERTER_KINDS.add(kind)


def unregister_converter(kind: str) -> None:
    _CONVERTER_KINDS.discard(kind)


def converter_kinds() -> frozenset[str]:
    return frozenset(_CONVERTER_KINDS)
```

`ProfileError` is defined below this point in the current file — move the `class ProfileError(ValueError)` definition above the registry block. In `__post_init__`, replace the `ALLOWED_CONVERTERS` check:

```python
        if self.converter not in _CONVERTER_KINDS:
            choices = ", ".join(sorted(_CONVERTER_KINDS))
            raise ProfileError(f"unknown converter {self.converter!r}; choices: {choices}")
```

Keep a module-level `ALLOWED_CONVERTERS = converter_kinds()` alias? No — grep for external users first: `grep -rn 'ALLOWED_CONVERTERS' src/ tests/ benchmarks/ --include='*.py' | grep -v __pycache__`. If any exist outside `profiles.py`, keep `ALLOWED_CONVERTERS` as a deprecated property returning `converter_kinds()`; if none, delete the name.

- [ ] **Step 4: Run the full profile test module**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_model_profiles.py -q`
Expected: PASS (all tests, not just the new ones)

- [ ] **Step 5: Commit**

```bash
git add src/mlx_vq/models/profiles.py tests/test_model_profiles.py
git commit -m "feat(profiles): open converter registry for new families

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Family adapter registry in ramp.models

`src/mlx_vq/models/__init__.py` is a hand-maintained export map where each family invents its own binder names. Add a registry that maps a family name to its adapter surface, so a new family registers instead of editing shared files. Physically lives in `src/ramp/models/registry.py` (cutover direction); `mlx_vq.models.registry` re-exports it.

**Files:**
- Create: `src/ramp/models/registry.py`
- Create: `src/mlx_vq/models/registry.py` (shim)
- Modify: `src/ramp/models/__init__.py`
- Test: `tests/test_family_registry.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `@dataclass(frozen=True) FamilyBinding(family: str, architecture: str, adapter_module: str, model_args_symbol: str, bind_vq_experts_symbol: str, bind_non_vq_weights_symbol: str | None, has_unbound_symbol: str, profile_name: str | None)`
  - `register_family(binding: FamilyBinding) -> None`
  - `get_family(family: str) -> FamilyBinding`
  - `list_families() -> tuple[str, ...]`
  - `resolve_family(family: str) -> ResolvedFamily` — imports the adapter module lazily and returns the callables
  - Registered defaults: `"glm52"`, `"glm45_air"`, `"qwen_moe"`. Task 6 registers `"deepseek_v4_flash"`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_family_registry.py
"""Family adapter registry: registration, lookup, lazy resolution."""

import pytest

from ramp.models import registry


def test_default_families_registered():
    families = registry.list_families()
    assert "glm52" in families
    assert "glm45_air" in families
    assert "qwen_moe" in families


def test_get_family_returns_binding():
    binding = registry.get_family("glm52")
    assert binding.architecture == "glm_moe_dsa"
    assert binding.adapter_module == "mlx_vq.models.glm52_vq_adapter"
    assert binding.bind_vq_experts_symbol == "bind_glm52_vq_experts"


def test_unknown_family_raises_with_choices():
    with pytest.raises(registry.FamilyRegistryError) as excinfo:
        registry.get_family("nope")
    assert "glm52" in str(excinfo.value)


def test_register_and_unregister_family():
    binding = registry.FamilyBinding(
        family="test_family",
        architecture="test_arch",
        adapter_module="mlx_vq.models.glm52_vq_adapter",
        model_args_symbol="GLM52VQModelArgs",
        bind_vq_experts_symbol="bind_glm52_vq_experts",
        bind_non_vq_weights_symbol="bind_glm52_non_vq_weights",
        has_unbound_symbol="has_unbound_vq_experts",
        profile_name=None,
    )
    registry.register_family(binding)
    try:
        assert "test_family" in registry.list_families()
    finally:
        registry.unregister_family("test_family")
    assert "test_family" not in registry.list_families()


def test_duplicate_registration_rejected():
    with pytest.raises(registry.FamilyRegistryError):
        registry.register_family(registry.get_family("glm52"))


def test_mlx_vq_shim_is_same_module():
    import mlx_vq.models.registry as shim

    assert shim.get_family is registry.get_family
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_family_registry.py -q`
Expected: FAIL with `ImportError: cannot import name 'registry' from 'ramp.models'`

- [ ] **Step 3: Implement `src/ramp/models/registry.py`**

```python
"""Registry mapping model families to their adapter surfaces.

Adding a family is a registration, not an edit to shared modules. Adapter
modules import MLX, so resolution is lazy: a binding stores dotted module
paths and symbol names; ``resolve_family`` imports on first use.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable


class FamilyRegistryError(ValueError):
    """Raised for unknown families or invalid registrations."""


@dataclass(frozen=True)
class FamilyBinding:
    family: str
    architecture: str
    adapter_module: str
    model_args_symbol: str
    bind_vq_experts_symbol: str
    bind_non_vq_weights_symbol: str | None
    has_unbound_symbol: str
    profile_name: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("family", "architecture", "adapter_module"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise FamilyRegistryError(f"{field_name} must be a non-empty string")


@dataclass(frozen=True)
class ResolvedFamily:
    binding: FamilyBinding
    model_args: type
    bind_vq_experts: Callable[..., Any]
    bind_non_vq_weights: Callable[..., Any] | None
    has_unbound: Callable[..., Any]


_REGISTRY: dict[str, FamilyBinding] = {}


def register_family(binding: FamilyBinding) -> None:
    if binding.family in _REGISTRY:
        raise FamilyRegistryError(f"family {binding.family!r} is already registered")
    _REGISTRY[binding.family] = binding


def unregister_family(family: str) -> None:
    _REGISTRY.pop(family, None)


def list_families() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def get_family(family: str) -> FamilyBinding:
    try:
        return _REGISTRY[family]
    except KeyError as error:
        choices = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise FamilyRegistryError(
            f"unknown family {family!r}; choices: {choices}"
        ) from error


def resolve_family(family: str) -> ResolvedFamily:
    binding = get_family(family)
    module = import_module(binding.adapter_module)
    bind_non_vq = (
        getattr(module, binding.bind_non_vq_weights_symbol)
        if binding.bind_non_vq_weights_symbol
        else None
    )
    return ResolvedFamily(
        binding=binding,
        model_args=getattr(module, binding.model_args_symbol),
        bind_vq_experts=getattr(module, binding.bind_vq_experts_symbol),
        bind_non_vq_weights=bind_non_vq,
        has_unbound=getattr(module, binding.has_unbound_symbol),
    )


register_family(
    FamilyBinding(
        family="glm52",
        architecture="glm_moe_dsa",
        adapter_module="mlx_vq.models.glm52_vq_adapter",
        model_args_symbol="GLM52VQModelArgs",
        bind_vq_experts_symbol="bind_glm52_vq_experts",
        bind_non_vq_weights_symbol="bind_glm52_non_vq_weights",
        has_unbound_symbol="has_unbound_vq_experts",
        profile_name="glm52-reap-504b-v2",
    )
)
register_family(
    FamilyBinding(
        family="glm45_air",
        architecture="glm4_moe",
        adapter_module="mlx_vq.models.glm45_air_vq_adapter",
        model_args_symbol="GLM45AirVQModel",
        bind_vq_experts_symbol="bind_glm45_air_vq_experts",
        bind_non_vq_weights_symbol="bind_glm45_air_non_expert_weights",
        has_unbound_symbol="has_unbound_glm45_air_vq_experts",
        profile_name="glm45-air",
    )
)
register_family(
    FamilyBinding(
        family="qwen_moe",
        architecture="qwen3_moe",
        adapter_module="mlx_vq.models.qwen_moe_adapter",
        model_args_symbol="QwenMoeBindReport",
        bind_vq_experts_symbol="bind_qwen_moe_vq_experts",
        bind_non_vq_weights_symbol="bind_qwen_non_expert_weights",
        has_unbound_symbol="has_unbound_qwen_moe_vq_experts",
        profile_name="qwen36-35b-a3b",
    )
)
```

Before committing, verify the three default bindings' symbol names against the actual adapter modules (`grep -n 'def bind_\|^class ' src/mlx_vq/models/glm45_air_vq_adapter.py src/mlx_vq/models/qwen_moe_adapter.py`) and correct `model_args_symbol` for glm45_air and qwen_moe to the real args/report classes those modules define — the names above were read from `mlx_vq/models/__init__.py`'s export map and the args-class choice for non-GLM52 families needs the module source as the authority.

- [ ] **Step 4: Create the shim `src/mlx_vq/models/registry.py`**

```python
"""Compatibility shim: canonical module is ``ramp.models.registry``."""

from ramp.models.registry import (  # noqa: F401
    FamilyBinding,
    FamilyRegistryError,
    ResolvedFamily,
    get_family,
    list_families,
    register_family,
    resolve_family,
    unregister_family,
)
```

- [ ] **Step 5: Export from `src/ramp/models/__init__.py`**

Append to the existing file (read it first; it is currently a near-empty alias stub — add without breaking existing content):

```python
from ramp.models import registry  # noqa: F401
```

- [ ] **Step 6: Run the tests**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_family_registry.py -q`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/ramp/models/registry.py src/mlx_vq/models/registry.py src/ramp/models/__init__.py tests/test_family_registry.py
git commit -m "feat(ramp): family adapter registry

New families register a FamilyBinding instead of editing the shared
models export map. Canonical home is ramp.models per the cutover;
mlx_vq.models.registry is the shim.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: DeepSeek-V4-Flash model profile

**Files:**
- Create: `models/deepseek-v4-flash-0731.yaml`
- Test: `tests/test_model_profiles.py`

**Interfaces:**
- Consumes: Task 3's `deepseek_v4_vq_groups` converter kind
- Produces: `get_profile("deepseek-v4-flash-0731")` works; Task 6's policy module names this profile

- [ ] **Step 1: Write the failing test** (append to `tests/test_model_profiles.py`)

```python
def test_deepseek_v4_flash_profile_loads():
    from mlx_vq.models.profiles import get_profile

    profile = get_profile("deepseek-v4-flash-0731")
    assert profile.hf_model_id == "deepseek-ai/DeepSeek-V4-Flash-0731"
    assert profile.architecture == "deepseek_v4"
    assert profile.num_layers == 43
    assert profile.hidden_size == 4096
    assert profile.vocab_size == 129280
    assert profile.num_experts == 256
    assert profile.experts_per_tok == 6
    assert profile.shared_experts == 1
    assert profile.moe_intermediate_size == 2048
    assert profile.converter == "deepseek_v4_vq_groups"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_model_profiles.py -q -k deepseek`
Expected: FAIL with `ProfileError: unknown model profile 'deepseek-v4-flash-0731'`

- [ ] **Step 3: Write the profile**

```yaml
# models/deepseek-v4-flash-0731.yaml
name: deepseek-v4-flash-0731
hf_model_id: deepseek-ai/DeepSeek-V4-Flash-0731
# Pin to an exact commit before any download-dependent work; null until then.
revision: null
architecture: deepseek_v4
num_layers: 43
# Dense-to-routed transition (first_k_dense_replace) is not confirmed from the
# released config yet; leave the sparse-layer split unset until measured.
num_sparse_layers: null
first_sparse_layer: null
hidden_size: 4096
moe_intermediate_size: 2048
num_experts: 256
experts_per_tok: 6
vocab_size: 129280
shared_experts: 1
converter: deepseek_v4_vq_groups
fused_gate_up: false
default_code_bits: 8
# Placeholder policy inherited from the GLM-5.2 ladder; must be re-derived for
# 256 experts before any materialization is accepted (pivot doc, open items).
group_size_policy:
  gate: 512
  up: 512
  down: 512
default_engine: vq_e1_routed_nax_e8
recovery_layer: 42
recovery_projections:
  - gate
  - up
  - down
hard_layers: []
imatrix_layers: []
# Source ships FP8 e4m3 (128x128 blocks, ue8m0 scales); ~284 GB native,
# ~568 GB dequantized to BF16. See keep-deepseek-v4-flash-pivot-20260811.md.
approx_bf16_gb: 568.0
notes: >
  DeepSeek-V4-Flash-0731. model_type deepseek_v4, MLA + indexer
  (index_topk 512, index_n_heads 64, index_head_dim 128), 1 MTP layer
  (num_nextn_predict_layers=1) which must be skipped or loaded
  intentionally. Source weights are block-FP8; dequantize via
  keep.convert.fp8_block before VQ fitting. Parameter count published
  inconsistently (284B vs 304B); measure from the shard index.
```

- [ ] **Step 4: Run the full profile test module**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_model_profiles.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add models/deepseek-v4-flash-0731.yaml tests/test_model_profiles.py
git commit -m "feat(models): DeepSeek-V4-Flash-0731 family profile

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: DeepSeek-V4 parameter policy and config validation

Mirror of `glm52_policy.py`'s job — classify every parameter path into a precision tier and validate an HF config — but written family-generically enough that the classifier handles the MTP layer by index. Lives under `src/ramp/models/`, shimmed from `mlx_vq.models`.

**Files:**
- Create: `src/ramp/models/deepseek_v4_policy.py`
- Create: `src/mlx_vq/models/deepseek_v4_policy.py` (shim)
- Modify: `src/ramp/models/registry.py` (register the family)
- Test: `tests/test_deepseek_v4_policy.py`

**Interfaces:**
- Consumes: Task 4's `register_family`; Task 5's profile
- Produces:
  - `class DeepseekV4Precision(str, Enum)` with values `vq_routed_expert`, `affine_or_fp16`, `fp16_or_fp32`, `router_fp16_or_affine_8_bit`, `mtp_skip`
  - `classify_deepseek_v4_parameter(path: str, *, num_hidden_layers: int = 43) -> DeepseekV4Precision`
  - `is_vq_routed_expert(path: str, *, num_hidden_layers: int = 43) -> bool`
  - `REQUIRED_DEEPSEEK_V4_CONFIG_FIELDS: tuple[str, ...]`
  - `validate_deepseek_v4_config(config: Mapping[str, object]) -> list[str]` returning failure tags, empty when valid

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_deepseek_v4_policy.py
"""DeepSeek-V4-Flash parameter classification and config validation."""

from ramp.models.deepseek_v4_policy import (
    DeepseekV4Precision,
    REQUIRED_DEEPSEEK_V4_CONFIG_FIELDS,
    classify_deepseek_v4_parameter,
    is_vq_routed_expert,
    validate_deepseek_v4_config,
)

VALID_CONFIG = {
    "model_type": "deepseek_v4",
    "architectures": ["DeepseekV4ForCausalLM"],
    "num_hidden_layers": 43,
    "hidden_size": 4096,
    "vocab_size": 129280,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "num_experts_per_tok": 6,
    "moe_intermediate_size": 2048,
    "q_lora_rank": 1024,
    "qk_rope_head_dim": 64,
    "max_position_embeddings": 1048576,
    "num_nextn_predict_layers": 1,
    "index_topk": 512,
    "index_n_heads": 64,
    "index_head_dim": 128,
    "quantization_config": {
        "quant_method": "fp8",
        "fmt": "e4m3",
        "scale_fmt": "ue8m0",
        "weight_block_size": [128, 128],
        "activation_scheme": "dynamic",
    },
}


def test_routed_expert_projections_are_vq():
    for proj in ("gate_proj", "up_proj", "down_proj"):
        path = f"model.layers.7.mlp.experts.13.{proj}.weight"
        assert (
            classify_deepseek_v4_parameter(path)
            is DeepseekV4Precision.VQ_ROUTED_EXPERT
        )
        assert is_vq_routed_expert(path)


def test_switch_mlp_paths_are_vq():
    path = "model.layers.7.mlp.switch_mlp.gate_proj.weight"
    assert (
        classify_deepseek_v4_parameter(path)
        is DeepseekV4Precision.VQ_ROUTED_EXPERT
    )


def test_router_shared_attn_and_embeddings():
    assert (
        classify_deepseek_v4_parameter("model.layers.7.mlp.gate.weight")
        is DeepseekV4Precision.ROUTER
    )
    assert (
        classify_deepseek_v4_parameter(
            "model.layers.7.mlp.shared_experts.gate_proj.weight"
        )
        is DeepseekV4Precision.AFFINE_OR_FP16
    )
    assert (
        classify_deepseek_v4_parameter(
            "model.layers.7.self_attn.indexer.wq_b.weight"
        )
        is DeepseekV4Precision.AFFINE_OR_FP16
    )
    assert (
        classify_deepseek_v4_parameter("model.embed_tokens.weight")
        is DeepseekV4Precision.FP16_OR_FP32
    )
    assert (
        classify_deepseek_v4_parameter("lm_head.weight")
        is DeepseekV4Precision.FP16_OR_FP32
    )


def test_mtp_layer_is_skip():
    # num_hidden_layers=43 means layer index 43 is the MTP head, not a
    # backbone layer; every tensor under it classifies as MTP_SKIP.
    path = "model.layers.43.mlp.experts.0.gate_proj.weight"
    assert (
        classify_deepseek_v4_parameter(path, num_hidden_layers=43)
        is DeepseekV4Precision.MTP_SKIP
    )
    assert not is_vq_routed_expert(path, num_hidden_layers=43)


def test_valid_config_passes():
    assert validate_deepseek_v4_config(VALID_CONFIG) == []


def test_missing_field_and_wrong_type_reported():
    config = dict(VALID_CONFIG)
    del config["index_topk"]
    config["model_type"] = "deepseek_v3"
    failures = validate_deepseek_v4_config(config)
    assert "missing_index_topk" in failures
    assert "model_type" in failures


def test_non_fp8_quantization_reported():
    config = dict(VALID_CONFIG)
    config["quantization_config"] = {"quant_method": "awq"}
    assert "quantization_config" in validate_deepseek_v4_config(config)


def test_required_fields_cover_moe_and_indexer():
    for field in ("n_routed_experts", "index_topk", "num_nextn_predict_layers"):
        assert field in REQUIRED_DEEPSEEK_V4_CONFIG_FIELDS


def test_family_registered():
    from ramp.models import registry

    binding = registry.get_family("deepseek_v4_flash")
    assert binding.architecture == "deepseek_v4"
    assert binding.profile_name == "deepseek-v4-flash-0731"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_deepseek_v4_policy.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ramp.models.deepseek_v4_policy'`

- [ ] **Step 3: Implement `src/ramp/models/deepseek_v4_policy.py`**

```python
"""Parameter precision policy and config validation for DeepSeek-V4-Flash.

Facts from deepseek-ai/DeepSeek-V4-Flash-0731 config.json, recorded in
keep-deepseek-v4-flash-pivot-20260811.md.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from enum import Enum

DEEPSEEK_V4_FLASH_MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash-0731"
DEEPSEEK_V4_NUM_HIDDEN_LAYERS = 43


class DeepseekV4Precision(str, Enum):
    VQ_ROUTED_EXPERT = "vq_routed_expert"
    AFFINE_OR_FP16 = "affine_or_fp16"
    FP16_OR_FP32 = "fp16_or_fp32"
    ROUTER = "router_fp16_or_affine_8_bit"
    MTP_SKIP = "mtp_skip"


REQUIRED_DEEPSEEK_V4_CONFIG_FIELDS = (
    "model_type",
    "architectures",
    "num_hidden_layers",
    "hidden_size",
    "vocab_size",
    "n_routed_experts",
    "n_shared_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
    "q_lora_rank",
    "qk_rope_head_dim",
    "max_position_embeddings",
    "num_nextn_predict_layers",
    "index_topk",
    "index_n_heads",
    "index_head_dim",
    "quantization_config",
)

_EXPECTED_SCALARS = {
    "model_type": "deepseek_v4",
    "num_hidden_layers": 43,
    "hidden_size": 4096,
    "vocab_size": 129280,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "num_experts_per_tok": 6,
    "moe_intermediate_size": 2048,
}

_LAYER_PATTERN = re.compile(r"\.layers\.(\d+)\.")


def _layer_index(path: str) -> int | None:
    match = _LAYER_PATTERN.search(path)
    return int(match.group(1)) if match else None


def classify_deepseek_v4_parameter(
    path: str,
    *,
    num_hidden_layers: int = DEEPSEEK_V4_NUM_HIDDEN_LAYERS,
) -> DeepseekV4Precision:
    layer = _layer_index(path)
    if layer is not None and layer >= num_hidden_layers:
        return DeepseekV4Precision.MTP_SKIP
    projection_markers = (".gate_proj.", ".up_proj.", ".down_proj.")
    if ".mlp.switch_mlp." in path and any(m in path for m in projection_markers):
        return DeepseekV4Precision.VQ_ROUTED_EXPERT
    if ".mlp.experts." in path and any(m in path for m in projection_markers):
        return DeepseekV4Precision.VQ_ROUTED_EXPERT
    if ".mlp.gate." in path:
        return DeepseekV4Precision.ROUTER
    if ".self_attn." in path:
        return DeepseekV4Precision.AFFINE_OR_FP16
    if ".mlp.shared_experts." in path:
        return DeepseekV4Precision.AFFINE_OR_FP16
    if any(m in path for m in ("embed_tokens", "lm_head", "norm", "layernorm")):
        return DeepseekV4Precision.FP16_OR_FP32
    if ".mlp." in path:
        return DeepseekV4Precision.AFFINE_OR_FP16
    return DeepseekV4Precision.FP16_OR_FP32


def is_vq_routed_expert(
    path: str,
    *,
    num_hidden_layers: int = DEEPSEEK_V4_NUM_HIDDEN_LAYERS,
) -> bool:
    return (
        classify_deepseek_v4_parameter(path, num_hidden_layers=num_hidden_layers)
        is DeepseekV4Precision.VQ_ROUTED_EXPERT
    )


def validate_deepseek_v4_config(config: Mapping[str, object]) -> list[str]:
    failures: list[str] = []
    for field in REQUIRED_DEEPSEEK_V4_CONFIG_FIELDS:
        if field not in config:
            failures.append(f"missing_{field}")
    for field, expected in _EXPECTED_SCALARS.items():
        if field in config and config[field] != expected:
            failures.append(field)
    quantization = config.get("quantization_config")
    if isinstance(quantization, Mapping):
        if quantization.get("quant_method") != "fp8":
            failures.append("quantization_config")
        elif quantization.get("fmt") != "e4m3":
            failures.append("quantization_config")
    elif "quantization_config" in config:
        failures.append("quantization_config")
    return list(dict.fromkeys(failures))
```

- [ ] **Step 4: Create the shim `src/mlx_vq/models/deepseek_v4_policy.py`**

```python
"""Compatibility shim: canonical module is ``ramp.models.deepseek_v4_policy``."""

from ramp.models.deepseek_v4_policy import (  # noqa: F401
    DEEPSEEK_V4_FLASH_MODEL_ID,
    DEEPSEEK_V4_NUM_HIDDEN_LAYERS,
    DeepseekV4Precision,
    REQUIRED_DEEPSEEK_V4_CONFIG_FIELDS,
    classify_deepseek_v4_parameter,
    is_vq_routed_expert,
    validate_deepseek_v4_config,
)
```

- [ ] **Step 5: Register the family** (append to the registration block in `src/ramp/models/registry.py`)

NOTE (post-Task-4): `src/ramp/models/registry.py` as landed is the authority
on the `FamilyBinding` signature — `model_args_symbol` is now `str | None`
and the earlier Task 4 snippets in this plan are stale. Register:

```python
register_family(
    FamilyBinding(
        family="deepseek_v4_flash",
        architecture="deepseek_v4",
        # Adapter module does not exist yet; the policy module carries the
        # family until the VQ adapter lands (follow-on plan). There is no
        # args class yet, so model_args_symbol stays None.
        adapter_module="ramp.models.deepseek_v4_policy",
        model_args_symbol=None,
        bind_vq_experts_symbol="classify_deepseek_v4_parameter",
        bind_non_vq_weights_symbol=None,
        has_unbound_symbol="is_vq_routed_expert",
        profile_name="deepseek-v4-flash-0731",
    )
)
```

- [ ] **Step 6: Run the tests**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_deepseek_v4_policy.py tests/test_family_registry.py -q`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/ramp/models/deepseek_v4_policy.py src/mlx_vq/models/deepseek_v4_policy.py src/ramp/models/registry.py tests/test_deepseek_v4_policy.py
git commit -m "feat(ramp): DeepSeek-V4-Flash parameter policy and config gate

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: FP8 block dequantization

DeepSeek-V4-Flash ships FP8 e4m3 weights in 128x128 blocks with ue8m0 scales. KEEP has no FP8 source path. Build a pure decode module, testable without the model: bit-exact e4m3 decode plus block-scale application.

**Files:**
- Create: `src/keep/convert/fp8_block.py`
- Test: `tests/test_fp8_block_dequant.py`

**Interfaces:**
- Consumes: nothing (numpy only; no MLX import so it runs headless)
- Produces:
  - `decode_e4m3(codes: np.ndarray) -> np.ndarray` — uint8 bit patterns to float32, handling sign/exponent(bias 7)/mantissa, subnormals, and NaN (S.1111.111)
  - `decode_ue8m0(scales: np.ndarray) -> np.ndarray` — uint8 exponent byte to float32 power of two, `2.0 ** (int(v) - 127)`
  - `dequantize_fp8_block(codes: np.ndarray, scales: np.ndarray, *, block_size: tuple[int, int] = (128, 128)) -> np.ndarray` — full weight dequant; `codes` shape `[O, I]` uint8, `scales` shape `[ceil(O/128), ceil(I/128)]` (uint8 ue8m0 or float32), returns float32 `[O, I]`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_fp8_block_dequant.py
"""Bit-exact FP8 e4m3 decode and 128x128 block dequantization."""

import numpy as np
import pytest

from keep.convert.fp8_block import (
    decode_e4m3,
    decode_ue8m0,
    dequantize_fp8_block,
)


def test_e4m3_known_values():
    # 0x00 = +0.0; 0x38 = 1.0 (exp 0b0111=bias, mantissa 0)
    # 0x3C = 1.5; 0xB8 = -1.0; 0x7E = 448.0 (max normal)
    codes = np.array([0x00, 0x38, 0x3C, 0xB8, 0x7E], dtype=np.uint8)
    expected = np.array([0.0, 1.0, 1.5, -1.0, 448.0], dtype=np.float32)
    np.testing.assert_array_equal(decode_e4m3(codes), expected)


def test_e4m3_subnormals():
    # exponent bits 0000, mantissa m: value = m/8 * 2**-6
    codes = np.array([0x01, 0x07], dtype=np.uint8)
    expected = np.array([2.0**-9, 7.0 / 8.0 * 2.0**-6], dtype=np.float32)
    np.testing.assert_allclose(decode_e4m3(codes), expected, rtol=0)


def test_e4m3_nan():
    # S.1111.111 is NaN in e4m3 (no infinities)
    values = decode_e4m3(np.array([0x7F, 0xFF], dtype=np.uint8))
    assert np.isnan(values).all()


def test_e4m3_roundtrip_against_ml_dtypes():
    ml_dtypes = pytest.importorskip("ml_dtypes")
    codes = np.arange(256, dtype=np.uint8)
    reference = codes.view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    ours = decode_e4m3(codes)
    both_nan = np.isnan(reference) & np.isnan(ours)
    np.testing.assert_array_equal(ours[~both_nan], reference[~both_nan])


def test_ue8m0_scale_decode():
    scales = np.array([127, 128, 126, 0], dtype=np.uint8)
    expected = np.array([1.0, 2.0, 0.5, 2.0**-127], dtype=np.float32)
    np.testing.assert_array_equal(decode_ue8m0(scales), expected)


def test_block_dequant_applies_per_block_scale():
    # 256x256 weight, 2x2 blocks of 128: every code is 1.0 (0x38),
    # scales double per block; output must equal the block scale.
    codes = np.full((256, 256), 0x38, dtype=np.uint8)
    scales = np.array([[127, 128], [129, 130]], dtype=np.uint8)
    weight = dequantize_fp8_block(codes, scales)
    assert weight.shape == (256, 256)
    assert weight.dtype == np.float32
    assert weight[0, 0] == 1.0
    assert weight[0, 255] == 2.0
    assert weight[255, 0] == 4.0
    assert weight[255, 255] == 8.0


def test_block_dequant_ragged_edge():
    # O and I not multiples of 128: last blocks are partial.
    codes = np.full((130, 200), 0x38, dtype=np.uint8)
    scales = np.full((2, 2), 128, dtype=np.uint8)
    weight = dequantize_fp8_block(codes, scales)
    assert weight.shape == (130, 200)
    assert (weight == 2.0).all()


def test_block_dequant_accepts_float_scales():
    codes = np.full((128, 128), 0x38, dtype=np.uint8)
    scales = np.array([[3.0]], dtype=np.float32)
    weight = dequantize_fp8_block(codes, scales)
    assert (weight == 3.0).all()


def test_block_dequant_rejects_wrong_scale_shape():
    codes = np.full((256, 256), 0x38, dtype=np.uint8)
    scales = np.full((1, 2), 127, dtype=np.uint8)
    with pytest.raises(ValueError):
        dequantize_fp8_block(codes, scales)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_fp8_block_dequant.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'keep.convert.fp8_block'`

Note: `keep.convert` is an alias package over `mlx_vq.convert` via `install_alias_package`. Check `src/keep/convert/__init__.py` first: if it aliases with `__path__` pointing at `mlx_vq/convert`, a new physical file in `src/keep/convert/` may be shadowed. If so, place the implementation at `src/keep/convert/fp8_block.py` AND verify import resolution; if the alias `__path__` wins, place the file at `src/mlx_vq/convert/fp8_block.py` instead and it will be importable under both names automatically. The test imports `keep.convert.fp8_block` either way — that import working is the acceptance criterion, the physical location follows from how the alias resolves.

- [ ] **Step 3: Implement**

```python
"""Decode FP8 e4m3 block-quantized weights (DeepSeek-V4 release format).

Format per config.json: quant_method fp8, fmt e4m3, scale_fmt ue8m0,
weight_block_size [128, 128], dynamic activation scheme. Weights are uint8
e4m3fn codes; each 128x128 block has one power-of-two scale stored as an
unsigned 8-bit exponent (ue8m0, bias 127).

Pure numpy so it runs headless; the MLX fast path can come later if decode
throughput matters.
"""

from __future__ import annotations

import numpy as np

_E4M3_TABLE: np.ndarray | None = None


def _e4m3_table() -> np.ndarray:
    global _E4M3_TABLE
    if _E4M3_TABLE is None:
        codes = np.arange(256, dtype=np.uint16)
        sign = np.where(codes & 0x80, -1.0, 1.0).astype(np.float64)
        exponent = ((codes >> 3) & 0x0F).astype(np.int64)
        mantissa = (codes & 0x07).astype(np.float64)
        normal = sign * (1.0 + mantissa / 8.0) * np.exp2(exponent - 7.0)
        subnormal = sign * (mantissa / 8.0) * np.exp2(-6.0)
        values = np.where(exponent == 0, subnormal, normal)
        # e4m3fn: S.1111.111 is NaN; there are no infinities.
        nan_mask = (exponent == 0x0F) & ((codes & 0x07) == 0x07)
        values = np.where(nan_mask, np.nan, values)
        _E4M3_TABLE = values.astype(np.float32)
    return _E4M3_TABLE


def decode_e4m3(codes: np.ndarray) -> np.ndarray:
    if codes.dtype != np.uint8:
        raise ValueError(f"e4m3 codes must be uint8, got {codes.dtype}")
    return _e4m3_table()[codes]


def decode_ue8m0(scales: np.ndarray) -> np.ndarray:
    if scales.dtype != np.uint8:
        raise ValueError(f"ue8m0 scales must be uint8, got {scales.dtype}")
    return np.exp2(scales.astype(np.float64) - 127.0).astype(np.float32)


def dequantize_fp8_block(
    codes: np.ndarray,
    scales: np.ndarray,
    *,
    block_size: tuple[int, int] = (128, 128),
) -> np.ndarray:
    if codes.ndim != 2:
        raise ValueError(f"codes must be 2-D, got shape {codes.shape}")
    block_rows, block_cols = block_size
    rows, cols = codes.shape
    expected = (-(-rows // block_rows), -(-cols // block_cols))
    if scales.shape != expected:
        raise ValueError(
            f"scales shape {scales.shape} does not match "
            f"{expected} blocks for weight {codes.shape} "
            f"at block size {block_size}"
        )
    if scales.dtype == np.uint8:
        scale_values = decode_ue8m0(scales)
    else:
        scale_values = scales.astype(np.float32)
    decoded = decode_e4m3(codes)
    expanded = np.repeat(
        np.repeat(scale_values, block_rows, axis=0), block_cols, axis=1
    )[:rows, :cols]
    return decoded * expanded
```

- [ ] **Step 4: Run the tests**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_fp8_block_dequant.py -q`
Expected: PASS (the ml_dtypes cross-check auto-skips if ml_dtypes is absent)

- [ ] **Step 5: Commit** (adjust the path if Step 2's alias check moved the file)

```bash
git add src/keep/convert/fp8_block.py tests/test_fp8_block_dequant.py
git commit -m "feat(keep): FP8 e4m3 block dequantization for DeepSeek-V4 sources

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: Parameterize the watchdog rule state in the GPU stack

The deployed `keep-glm52-sky-watchdog` EventBridge rule was disabled by hand on 2026-08-11 when the GLM-5.2 run was killed, but `aws/glm52-gpu/cfn/gpu-teacher-stack.yaml:2153` hardcodes `State: ENABLED` — the next stack update would silently re-enable a watchdog pointed at a dead campaign. Make the state a parameter defaulting to `DISABLED`.

**Files:**
- Modify: `aws/glm52-gpu/cfn/gpu-teacher-stack.yaml` (Parameters section, and line 2153 inside the `SkyWatchdogRule` resource at lines 2146-2153)
- Test: `tests/test_gpu_teacher_stack_template.py`

**Interfaces:**
- Consumes: nothing
- Produces: `SkyWatchdogRuleState` template parameter

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gpu_teacher_stack_template.py
"""Text-level contract tests for the GPU teacher CloudFormation template.

The template uses CFN YAML short tags (!Ref, !GetAtt) that plain yaml.safe_load
rejects, so assertions are textual.
"""

import re
from pathlib import Path

TEMPLATE = Path("aws/glm52-gpu/cfn/gpu-teacher-stack.yaml").read_text()


def test_watchdog_rule_state_is_parameterized():
    assert "SkyWatchdogRuleState:" in TEMPLATE
    parameter_block = TEMPLATE.split("SkyWatchdogRuleState:", 1)[1][:400]
    assert "Default: DISABLED" in parameter_block
    assert "ENABLED" in parameter_block  # allowed value


def test_watchdog_rule_references_the_parameter():
    match = re.search(
        r"SkyWatchdogRule:\n(?:.*\n)*?      State: (.+)", TEMPLATE
    )
    assert match is not None, "SkyWatchdogRule resource not found"
    assert match.group(1).strip() == "!Ref SkyWatchdogRuleState"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_gpu_teacher_stack_template.py -q`
Expected: FAIL on the first assertion

- [ ] **Step 3: Edit the template**

In the `Parameters:` section of `aws/glm52-gpu/cfn/gpu-teacher-stack.yaml`, add (match the file's existing parameter indentation):

```yaml
  SkyWatchdogRuleState:
    Type: String
    AllowedValues:
      - ENABLED
      - DISABLED
    Default: DISABLED
    Description: >-
      State of the campaign watchdog schedule. DISABLED since the GLM-5.2
      retirement (2026-08-11); set ENABLED when a live campaign needs it.
```

In the `SkyWatchdogRule` resource (the block starting at line 2146 — verify it is the resource whose `Name` output is `SkyWatchdogRuleName`, not one of the other `State: ENABLED` rules at lines 1363/1425/1824), change:

```yaml
      State: ENABLED
```

to:

```yaml
      State: !Ref SkyWatchdogRuleState
```

- [ ] **Step 4: Run the test**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_gpu_teacher_stack_template.py -q`
Expected: PASS

- [ ] **Step 5: Verify no other template tests broke**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests -q -k "template or stack" 2>&1 | tail -5`
Expected: no new failures

- [ ] **Step 6: Commit**

```bash
git add aws/glm52-gpu/cfn/gpu-teacher-stack.yaml tests/test_gpu_teacher_stack_template.py
git commit -m "fix(aws): parameterize watchdog rule state, default DISABLED

The deployed rule was disabled at GLM-5.2 retirement; the hardcoded
ENABLED would have re-armed it on the next stack update.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

Do NOT deploy this stack update as part of this plan — the deployed rule is already disabled; the template change just settles the drift whenever the stack next updates for real work.

---

### Task 9: Point the docs at the new family

**Files:**
- Modify: `README.md` (add a short status note near the top pointing at the pivot doc)
- Modify: `docs/KEEP_NEW_MODEL_FAMILY_TEMPLATE.md` (the sentence "Existing `mlx_vq.*` imports remain supported as the Air-ladder compatibility shim until the GLM-5.2 cutover.")

**Interfaces:**
- Consumes: Task 2's pivot doc
- Produces: docs consistent with the executed pivot

- [ ] **Step 1: Edit README.md**

Read the README's opening section first, then insert after the title/intro paragraph (match surrounding tone):

```markdown
> **Model target (2026-08-11):** KEEP's primary family is now
> `deepseek-ai/DeepSeek-V4-Flash-0731`. GLM-5.2 is retired; see
> [keep-deepseek-v4-flash-pivot-20260811.md](keep-deepseek-v4-flash-pivot-20260811.md)
> for the decision record and shutdown evidence.
```

- [ ] **Step 2: Edit the template doc**

Replace the sentence:

```text
Existing `mlx_vq.*` imports remain supported as the Air-ladder
compatibility shim until the GLM-5.2 cutover.
```

with:

```text
Existing `mlx_vq.*` imports remain supported as a compatibility shim.
The cutover began 2026-08-11 with the DeepSeek-V4-Flash pivot: new
family code lives in `ramp.models` / `keep.convert` and is registered
through `ramp.models.registry` instead of edited into shared modules.
```

- [ ] **Step 3: Commit**

```bash
git add README.md docs/KEEP_NEW_MODEL_FAMILY_TEMPLATE.md
git commit -m "docs: point README and family template at DeepSeek pivot

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 10: Full-suite verification

**Files:** none

**Interfaces:**
- Consumes: every prior task
- Produces: evidence the cutover changes are regression-free

- [ ] **Step 1: Run the model-layer and new test modules**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests/test_model_profiles.py tests/test_family_registry.py tests/test_deepseek_v4_policy.py tests/test_fp8_block_dequant.py tests/test_gpu_teacher_stack_template.py tests/test_glm52_policy.py tests/test_glm52_vq_adapter.py -q`
Expected: PASS

- [ ] **Step 2: Run the full suite**

Run: `UV_CACHE_DIR=/tmp/keep-uv-cache uv run --group dev python -m pytest tests -q 2>&1 | tail -5`
Expected: no failures beyond any that pre-exist on `main` (establish the baseline first if unsure: `git stash && pytest ... && git stash pop` is NOT needed — all work is committed by now; instead compare against the failure list from before Task 3 if any test was already red)

- [ ] **Step 3: Report**

Summarize: tasks completed, test counts, any deviations from plan.

---

## Explicitly out of scope (follow-on plans)

1. **DeepSeek-V4-Flash VQ adapter** (`ramp/models/deepseek_v4_vq_adapter.py`): model class over mlx-lm's `deepseek_v32` attention (mlx-lm 0.31.3 has no `deepseek_v4` module; the `deepseek_v32` `ModelArgs` field set matches V4's config field-for-field), MTP skip, binder implementations satisfying the Task 4 registry contract. Requires downloaded weights to pin tensor names.
2. **Weight download and measurement**: pin `revision`, measure true parameter count (284 B vs 304 B), record `first_k_dense_replace`, update the profile's `num_sparse_layers`/`first_sparse_layer`/`revision`.
3. **`src/glm52_enforcement/` generalization** (120 modules of campaign/spend/fence machinery): rename to a model-neutral package with per-campaign model identity as data, not code. Large, mechanical, deserves its own plan after the DeepSeek campaign shape is known.
4. **Teacher-gen campaign for DeepSeek**: new run id, new descriptors, capacity-block acquisition (`describe-capacity-block-offerings`, not on-demand retries — see pivot doc), corpus reuse of the retained teich pack.
5. **Group-size policy and codebook re-derivation** for 256-expert, 2048-wide experts.
