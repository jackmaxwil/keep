"""Deterministic schedule and durable training-state tests."""

from __future__ import annotations

import json

import numpy as np
import pytest

from mlx_vq.quality.glm52_teich_training_campaign import (
    TrainingCheckpointConfig,
    TrainingCheckpointStore,
    TrainingIdentity,
    ValidationState,
    build_fixed_split_windows,
    build_training_schedule,
    run_numpy_sgd_schedule,
)


def _sha(value: str) -> str:
    return value * 64


def _manifest_entries() -> list[dict[str, object]]:
    return [
        {
            "prompt_id": "train-a",
            "split": "train",
            "tuning_eligible": True,
            "supervised_position_count": 130,
        },
        {
            "prompt_id": "validation-a",
            "split": "validation",
            "tuning_eligible": False,
            "supervised_position_count": 70,
        },
        {
            "prompt_id": "holdout-a",
            "split": "holdout",
            "tuning_eligible": False,
            "supervised_position_count": 99,
        },
        {
            "prompt_id": "train-b",
            "split": "train",
            "tuning_eligible": True,
            "supervised_position_count": 17,
        },
    ]


def _identity(schedule_hash: str, *, baseline: str | None = None) -> TrainingIdentity:
    return TrainingIdentity(
        run_id="teich-train-20260717",
        baseline_sha256=baseline or _sha("1"),
        teacher_manifest_sha256=_sha("2"),
        schedule_sha256=schedule_hash,
        training_config_sha256=_sha("3"),
    )


def test_schedule_is_deterministic_one_epoch_and_excludes_non_train() -> None:
    first = build_training_schedule(_manifest_entries(), window_size=64, seed=20260712)
    second = build_training_schedule(_manifest_entries(), window_size=64, seed=20260712)

    assert first == second
    assert first.sha256 == second.sha256
    assert len(first.windows) == 4
    assert {window.prompt_id for window in first.windows} == {"train-a", "train-b"}
    covered = sorted(
        (window.prompt_id, position)
        for window in first.windows
        for position in range(window.start, window.end)
    )
    expected = sorted(
        [("train-a", i) for i in range(130)]
        + [("train-b", i) for i in range(17)]
    )
    assert covered == expected
    assert all(0 < window.end - window.start <= 64 for window in first.windows)
    validation = build_fixed_split_windows(
        _manifest_entries(), split="validation", max_windows=2
    )
    assert {
        (window.prompt_id, window.start, window.end) for window in validation
    } == {
        ("validation-a", 0, 64),
        ("validation-a", 64, 70),
    }


def test_training_checkpoint_roundtrip_and_marker_failure(tmp_path, monkeypatch) -> None:
    schedule = build_training_schedule(_manifest_entries())
    store = TrainingCheckpointStore(tmp_path, identity=_identity(schedule.sha256))
    first = store.publish(
        params={"left": np.ones((2, 2), dtype=np.float32)},
        completed_schedule_index=1,
        losses=np.array([1.5], dtype=np.float32),
        latest_gradient_norms={"left": 0.25},
        best_validation_metric=0.8,
        best_checkpoint_sha256=None,
        consecutive_validation_regressions=0,
        rng_state={"seed": 7, "offset": 1},
    )

    def fail_latest(_payload) -> None:
        raise OSError("simulated training marker failure")

    monkeypatch.setattr(store, "_publish_latest", fail_latest)
    with pytest.raises(OSError, match="training marker"):
        store.publish(
            params={"left": np.full((2, 2), 2.0, dtype=np.float32)},
            completed_schedule_index=2,
            losses=np.array([1.5, 1.0], dtype=np.float32),
            latest_gradient_norms={"left": 0.1},
            best_validation_metric=0.7,
            best_checkpoint_sha256=first.checkpoint_sha256,
            consecutive_validation_regressions=0,
            rng_state={"seed": 7, "offset": 2},
        )

    restored = TrainingCheckpointStore(
        tmp_path, identity=_identity(schedule.sha256)
    ).load_latest()
    assert restored.completed_schedule_index == 1
    np.testing.assert_array_equal(restored.params["left"], np.ones((2, 2), np.float32))
    with pytest.raises(ValueError, match="identity"):
        TrainingCheckpointStore(
            tmp_path,
            identity=_identity(schedule.sha256, baseline=_sha("4")),
        ).load_latest()


def test_interrupted_and_resumed_numpy_sgd_matches_uninterrupted(tmp_path) -> None:
    schedule = build_training_schedule(_manifest_entries())
    identity = _identity(schedule.sha256)
    initial = {"weight": np.array([0.3, -0.7], dtype=np.float32)}

    def gradient(params, window, _step):
        target = np.float32((window.end - window.start) / 100.0)
        return {"weight": params["weight"] - target}

    uninterrupted = run_numpy_sgd_schedule(
        initial_params=initial,
        schedule=schedule,
        learning_rate=0.2,
        gradient_fn=gradient,
    )
    store = TrainingCheckpointStore(tmp_path, identity=identity)
    partial = run_numpy_sgd_schedule(
        initial_params=initial,
        schedule=schedule,
        learning_rate=0.2,
        gradient_fn=gradient,
        checkpoint_store=store,
        checkpoint_every_steps=1,
        stop_after_steps=2,
    )
    assert partial.completed_schedule_index == 2
    resumed = run_numpy_sgd_schedule(
        initial_params=initial,
        schedule=schedule,
        learning_rate=0.2,
        gradient_fn=gradient,
        checkpoint_store=store,
        checkpoint_every_steps=1,
        resume=True,
    )

    np.testing.assert_array_equal(resumed.params["weight"], uninterrupted.params["weight"])
    np.testing.assert_array_equal(resumed.losses, uninterrupted.losses)


def test_training_store_recovers_authenticated_best_validation_params(tmp_path) -> None:
    schedule = build_training_schedule(_manifest_entries())
    store = TrainingCheckpointStore(tmp_path, identity=_identity(schedule.sha256))
    best = store.publish(
        params={"left": np.ones((2, 2), dtype=np.float32)},
        completed_schedule_index=1,
        losses=np.array([1.0], dtype=np.float32),
        latest_gradient_norms={"left": 0.2},
        best_validation_metric=0.5,
        best_checkpoint_sha256=None,
        consecutive_validation_regressions=0,
        rng_state={"offset": 1},
        best_is_current=True,
    )
    store.publish(
        params={"left": np.full((2, 2), 2.0, dtype=np.float32)},
        completed_schedule_index=2,
        losses=np.array([1.0, 1.1], dtype=np.float32),
        latest_gradient_norms={"left": 0.3},
        best_validation_metric=0.5,
        best_checkpoint_sha256=best.checkpoint_sha256,
        consecutive_validation_regressions=1,
        rng_state={"offset": 2},
    )

    restored = store.load_best_params()
    np.testing.assert_array_equal(restored["left"], np.ones((2, 2), dtype=np.float32))


def test_validation_state_retains_best_and_stops_after_two_regressions() -> None:
    state = ValidationState()
    state = state.observe(metric=1.0, checkpoint_sha256=_sha("1"))
    assert state.best_metric == 1.0 and not state.should_stop
    state = state.observe(metric=0.8, checkpoint_sha256=_sha("2"))
    assert state.best_metric == 0.8 and state.consecutive_regressions == 0
    state = state.observe(metric=0.9, checkpoint_sha256=_sha("3"))
    assert state.consecutive_regressions == 1 and not state.should_stop
    state = state.observe(metric=0.95, checkpoint_sha256=_sha("4"))
    assert state.should_stop


def test_checkpoint_config_canonical_campaign_defaults(tmp_path) -> None:
    config = TrainingCheckpointConfig(local_checkpoint_dir=tmp_path)
    assert config.local_every_steps == 50
    assert config.s3_every_steps == 250
    assert config.s3_every_seconds == 300
    assert config.validation_every_steps == 2048
    payload = json.loads(config.canonical_candidate_json())
    assert payload["layers"] == [77]
    assert payload["projections"] == ["gate_proj", "up_proj", "down_proj"]
    assert payload["rank"] == 4
    assert payload["learning_rate"] == 0.2
    assert payload["top_k"] == 2048
