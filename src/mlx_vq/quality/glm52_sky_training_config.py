"""Immutable canonical training configuration for the Sky campaign."""

from __future__ import annotations

import hashlib
import json
from typing import Mapping


_BODY = {
    "schema_version": 1,
    "record_type": "glm52_sky_training_configuration_v1",
    "seed": 20260712,
    "window_size": 64,
    "max_epochs": 1,
    "validation_every_steps": 2048,
    "stop_after_validation_regressions": 2,
    "checkpoint_every_steps": 50,
    "s3_sync_every_steps": 250,
    "s3_sync_every_seconds": 300,
    "layer": 77,
    "projections": ["gate_proj", "up_proj", "down_proj"],
    "rank": 4,
    "learning_rate": 0.2,
    "initialization_scale": 0.02,
    "gradient_clip_norm": 1.0,
    "topk_kl_enabled": True,
    "topk": 2048,
    "cka_enabled": True,
    "cka_layers": [77],
    "cka_balance": "dynamic_kl_relative",
    "router_kl_weight": 1.0,
    "router_entropy_beta": 0.01,
    "monte_carlo_expert_exploration": True,
    "target_nll_weight": 0.0,
    "top1_margin_weight": 0.0,
    "expansion_layers": [75, 74, 76, 72, 73, 71, 70],
    "automatic_promotion": False,
    "model_upload_authorized": False,
}
_FIELDS = frozenset(_BODY) | {"training_config_body_sha256"}


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_sky_training_config() -> dict[str, object]:
    body = dict(_BODY)
    body["projections"] = list(_BODY["projections"])
    body["cka_layers"] = list(_BODY["cka_layers"])
    body["expansion_layers"] = list(_BODY["expansion_layers"])
    return {**body, "training_config_body_sha256": _sha(body)}


def validate_sky_training_config(value: Mapping[str, object]) -> dict[str, object]:
    if set(value) != _FIELDS:
        raise ValueError("Sky training configuration schema mismatch")
    body = dict(value)
    digest = body.pop("training_config_body_sha256")
    if digest != _sha(body):
        raise ValueError("Sky training configuration body SHA-256 mismatch")
    if body != _BODY:
        raise ValueError("Sky training configuration authority mismatch")
    return dict(value)
