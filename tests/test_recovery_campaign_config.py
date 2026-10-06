from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml

from mlx_vq.recovery_campaign import (
    CampaignConfigError,
    CampaignObservation,
    load_campaign_config,
)


RECIPE = Path("recipes/glm52_recovery_campaign_v1_20260711.yaml")


def _raw_recipe() -> dict[str, object]:
    payload = yaml.safe_load(RECIPE.read_text())
    assert isinstance(payload, dict)
    return payload


def _write_recipe(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


def _write_raw_recipe(tmp_path: Path, payload: str) -> Path:
    path = tmp_path / "campaign.yaml"
    path.write_text(payload)
    return path


def _transition_raw(raw: dict[str, object], name: str) -> dict[str, object]:
    return next(item for item in raw["transitions"] if item["name"] == name)


def _authority_raw(raw: dict[str, object], name: str) -> dict[str, object]:
    return next(item for item in raw["authorities"] if item["name"] == name)


def _replace_option(argv: list[str], flag: str, value: str) -> None:
    index = argv.index(flag)
    argv[index + 1] = value


def _remove_option(argv: list[str], flag: str, *, has_value: bool = True) -> None:
    index = argv.index(flag)
    del argv[index : index + (2 if has_value else 1)]


def test_tracked_campaign_binds_current_authorities_and_canonical_budget() -> None:
    config = load_campaign_config(RECIPE)

    assert config.campaign == "glm52-recovery-v1-20260711"
    assert config.model_id == "0xSero/glm-5.2-reap-504B-v2"
    assert config.model_revision == "6c9241aa05fb243a0edb7c804c213ec1cf5c920d"
    assert config.tuning.split == "selection"
    assert (config.tuning.prompt_count, config.tuning.position_count) == (22, 255)
    assert tuple(experiment.name for experiment in config.experiments) == (
        "full75-e8",
        "worst8-e8p",
        "ebss",
        "rotations",
    )

    authorities = {authority.name: authority for authority in config.authorities}
    assert authorities["recovery-stats"].sha256 == (
        "9c743281c358332b4ff473d87e17b796275007f4662cc9363cbfd59da56196d7"
    )
    assert authorities["layer-attribution"].sha256 == (
        "81986872f6243e30739306868fbb7316fd7f2da6135740155e0e22de884f0b88"
    )
    assert authorities["full75-recovered-attribution"].path == (
        "artifacts/quality/glm52-recovery-wave1-reattribution75-20260711.json"
    )
    assert authorities["full75-recovered-attribution"].sha256 == (
        "f4db48e00e32b6b1df502bfa7e91902b3a9c008160dd45285d49a66e97d67ee8"
    )
    assert authorities["accepted-seed-manifest"].sha256 == (
        "ba1d3135ef8901f1a69ead28b5f9d330ef40d015ac31fdb4d2dcfe678c41f0a4"
    )
    assert authorities["accepted-composite-audit"].sha256 == (
        "8026322a533606ed451d13b029d83836fefbf1a938b33529de535f3fc778591a"
    )
    assert authorities["family-prompt-pack"].role == "artifact"
    assert authorities["family-prompt-pack"].split is None
    assert config.experiment("full75-e8").tuning_authorities == (
        "recovery-stats",
        "layer-attribution",
    )
    assert all(
        config.experiment(name).tuning_authorities
        == ("recovery-stats", "full75-recovered-attribution")
        for name in ("worst8-e8p", "ebss", "rotations")
    )

    worst8 = config.experiment("worst8-e8p")
    assert tuple(rate.layer for rate in worst8.projection_rates) == (
        75,
        74,
        76,
        72,
        73,
        77,
        71,
        70,
    )
    assert all(rate.values == (16, 16, 16) for rate in worst8.projection_rates)
    assert worst8.expected_payload_bytes == 104_775_711_456
    assert config.canonical_payload_bytes(worst8) == worst8.expected_payload_bytes
    assert config.canonical_payload_bytes(config.experiment("full75-e8")) == (
        98_433_923_808
    )

    full75 = config.transition("full75-rematerialize")
    assert full75.heavy is True
    assert full75.argv[:3] == (
        ".venv/bin/python",
        "benchmarks/run_glm52_recovery_wave1.py",
        "rematerialize",
    )
    assert "GLM_MLX_WIRED_LIMIT_GB" not in " ".join(full75.argv)


def test_tracked_campaign_binds_typed_model_source_authority() -> None:
    config = load_campaign_config(RECIPE)

    assert config.source_dir == (
        "/Users/jack.mazac/.cache/huggingface/hub/"
        "models--0xSero--glm-5.2-reap-504B-v2/snapshots/"
        "6c9241aa05fb243a0edb7c804c213ec1cf5c920d"
    )
    assert config.index_path == f"{config.source_dir}/model.safetensors.index.json"
    assert config.full_source_blob_inventory_sha256 == (
        "ace08e87dcbce3a22249e54196a27c0045992f8d5b8ca07f342899fa7a53fc8d"
    )
    assert config.routed_source_blob_inventory_sha256 == (
        "5dbacc9b0ec3deee829027fe739ebca9d0a77e8cd0016977b799e3a52539e2da"
    )
    assert config.config_sha256 == (
        "5fa690755d0dab25a8e0e5e0745675bdac03ba2b6f5641da2931278235c71f1b"
    )
    assert config.index_sha256 == (
        "bb5b4fa9782aea5ffc66f9145d6e630f1045d385c30437c531bebe422c075f3f"
    )
    assert Path(config.source_dir).name == config.model_revision


def test_model_revision_must_name_source_snapshot(tmp_path: Path) -> None:
    raw = _raw_recipe()
    raw["model"]["revision"] = "0" * 40

    with pytest.raises(CampaignConfigError, match="revision.*source"):
        load_campaign_config(_write_recipe(tmp_path, raw))


def test_source_dir_must_bind_model_id_cache_component(tmp_path: Path) -> None:
    raw = _raw_recipe()
    revision = raw["model"]["revision"]
    source_dir = f"/tmp/models--wrong--model/snapshots/{revision}"
    index_path = f"{source_dir}/model.safetensors.index.json"
    raw["model"]["source_dir"] = source_dir
    raw["model"]["index_path"] = index_path
    for transition in raw["transitions"]:
        _replace_option(transition["argv"], "--source-dir", source_dir)
        _replace_option(transition["argv"], "--index-path", index_path)

    with pytest.raises(CampaignConfigError, match="model_id.*source_dir"):
        load_campaign_config(_write_recipe(tmp_path, raw))


def test_index_path_is_exact_source_index_filename(tmp_path: Path) -> None:
    raw = _raw_recipe()
    index_path = f"{raw['model']['source_dir']}/other-index.json"
    raw["model"]["index_path"] = index_path
    for transition in raw["transitions"]:
        _replace_option(transition["argv"], "--index-path", index_path)

    with pytest.raises(CampaignConfigError, match="model.safetensors.index.json"):
        load_campaign_config(_write_recipe(tmp_path, raw))


def test_domain_values_are_immutable() -> None:
    config = load_campaign_config(RECIPE)

    with pytest.raises(FrozenInstanceError):
        config.experiments[0].name = "changed"  # type: ignore[misc]

    observation = CampaignObservation(
        lock_held=False,
        lock_owner_known=False,
        active_processes=(),
        recovered_groups=0,
        expected_groups=225,
        manifest_sha256=None,
        contradictions=(),
    )
    with pytest.raises(FrozenInstanceError):
        observation.lock_held = True  # type: ignore[misc]


@pytest.mark.parametrize(
    "mutate",
    (
        lambda raw: raw.update({"unexpected": True}),
        lambda raw: raw["authorities"][0].update({"unexpected": True}),
        lambda raw: raw["experiments"][0].update({"unexpected": True}),
    ),
)
def test_unknown_keys_fail_closed(tmp_path: Path, mutate) -> None:
    raw = _raw_recipe()
    mutate(raw)

    with pytest.raises(CampaignConfigError, match="unknown fields"):
        load_campaign_config(_write_recipe(tmp_path, raw))


def test_duplicate_experiment_names_fail_closed(tmp_path: Path) -> None:
    raw = _raw_recipe()
    raw["experiments"][1]["name"] = raw["experiments"][0]["name"]

    with pytest.raises(CampaignConfigError, match="duplicate experiment"):
        load_campaign_config(_write_recipe(tmp_path, raw))


@pytest.mark.parametrize(
    "mutate",
    (
        lambda raw: raw["tuning"].update({"split": "holdout"}),
        lambda raw: raw["tuning"].update({"holdout_used_for_tuning": True}),
        lambda raw: raw["authorities"][0].update({"split": "report"}),
    ),
)
def test_only_selection_evidence_may_tune(tmp_path: Path, mutate) -> None:
    raw = _raw_recipe()
    mutate(raw)

    with pytest.raises(CampaignConfigError, match="selection"):
        load_campaign_config(_write_recipe(tmp_path, raw))


def test_rate_override_requires_gate_up_down_triplet(tmp_path: Path) -> None:
    raw = _raw_recipe()
    worst8 = raw["experiments"][1]
    worst8["projection_rates"]["77"].pop("down")

    with pytest.raises(CampaignConfigError, match="gate.*up.*down"):
        load_campaign_config(_write_recipe(tmp_path, raw))


def test_rate_override_requires_uniform_whole_layer_triplet(tmp_path: Path) -> None:
    raw = _raw_recipe()
    ebss = raw["experiments"][2]
    ebss["projection_rates"]["77"]["up"] = 8
    ebss["expected_payload_bytes"] -= raw["budget"][
        "e8p_increment_per_projection_bytes"
    ]

    with pytest.raises(CampaignConfigError, match="uniform.*8.*16"):
        load_campaign_config(_write_recipe(tmp_path, raw))


@pytest.mark.parametrize(
    "unsafe_path",
    (
        "/tmp/glm52-campaign-output",
        "../glm52-campaign-output",
        "runs/glm52-campaign-output",
        ".keep-heavy-job.lock",
    ),
)
def test_experiment_outputs_must_be_safe_artifact_paths(
    tmp_path: Path, unsafe_path: str
) -> None:
    raw = _raw_recipe()
    raw["experiments"][0]["output_path"] = unsafe_path

    with pytest.raises(CampaignConfigError, match="safe artifact path"):
        load_campaign_config(_write_recipe(tmp_path, raw))


@pytest.mark.parametrize("failure", ("declared", "limit"))
def test_payload_budget_math_fails_closed(tmp_path: Path, failure: str) -> None:
    raw = _raw_recipe()
    worst8 = raw["experiments"][1]
    if failure == "declared":
        worst8["expected_payload_bytes"] += 1
        message = "canonical payload"
    else:
        raw["budget"]["payload_limit_bytes"] = worst8["expected_payload_bytes"] - 1
        message = "payload limit"

    with pytest.raises(CampaignConfigError, match=message):
        load_campaign_config(_write_recipe(tmp_path, raw))


@pytest.mark.parametrize(
    "output_path",
    (
        "artifacts/quality/a-different-candidate",
        "runs/glm52-recovery-candidate",
    ),
)
def test_rematerialize_output_option_must_match_experiment(
    tmp_path: Path, output_path: str
) -> None:
    raw = _raw_recipe()
    transition = _transition_raw(raw, "full75-rematerialize")
    _replace_option(transition["argv"], "--output-dir", output_path)

    with pytest.raises(CampaignConfigError, match="--output-dir"):
        load_campaign_config(_write_recipe(tmp_path, raw))


@pytest.mark.parametrize("failure", ("not-heavy", "missing", "wrong"))
def test_rematerialize_requires_exact_heavy_lock(
    tmp_path: Path, failure: str
) -> None:
    raw = _raw_recipe()
    transition = _transition_raw(raw, "full75-rematerialize")
    argv = transition["argv"]
    if failure == "not-heavy":
        transition["heavy"] = False
    elif failure == "missing":
        index = argv.index("--heavy-lock-path")
        del argv[index : index + 2]
    else:
        _replace_option(argv, "--heavy-lock-path", "artifacts/quality/not-the-lock")

    with pytest.raises(CampaignConfigError, match="heavy|--heavy-lock-path"):
        load_campaign_config(_write_recipe(tmp_path, raw))


@pytest.mark.parametrize(
    ("authority_name", "flag", "use_hash"),
    (
        ("recovery-stats", "--stats-dir", False),
        ("recovery-stats", "--expected-stats-manifest-sha256", True),
        ("layer-attribution", "--attribution-json", False),
        ("layer-attribution", "--expected-attribution-sha256", True),
        ("accepted-seed-manifest", "--seed-artifact-dir", False),
        ("accepted-seed-manifest", "--expected-seed-manifest-sha256", True),
        ("accepted-composite-audit", "--accepted-composite-audit-json", False),
        ("accepted-composite-audit", "--expected-composite-audit-sha256", True),
    ),
)
def test_rematerialize_options_must_close_over_authorities(
    tmp_path: Path, authority_name: str, flag: str, use_hash: bool
) -> None:
    raw = _raw_recipe()
    transition = _transition_raw(raw, "full75-rematerialize")
    value = "0" * 64 if use_hash else "artifacts/quality/wrong-authority"
    _replace_option(transition["argv"], flag, value)

    with pytest.raises(CampaignConfigError, match=f"{flag}|{authority_name}"):
        load_campaign_config(_write_recipe(tmp_path, raw))


def test_transition_binds_experiment_selected_attribution_authority(
    tmp_path: Path,
) -> None:
    raw = _raw_recipe()
    recovered_attribution = deepcopy(_authority_raw(raw, "layer-attribution"))
    recovered_attribution.update(
        {
            "name": "alternate-recovered-attribution",
            "path": "artifacts/quality/glm52-recovery-full75-attribution.json",
            "sha256": "1" * 64,
        }
    )
    raw["authorities"].append(recovered_attribution)
    worst8 = raw["experiments"][1]
    worst8["tuning_authorities"] = [
        "recovery-stats",
        "alternate-recovered-attribution",
    ]
    transition = _transition_raw(raw, "worst8-e8p-rematerialize")
    _replace_option(
        transition["argv"],
        "--attribution-json",
        recovered_attribution["path"],
    )
    _replace_option(
        transition["argv"],
        "--expected-attribution-sha256",
        recovered_attribution["sha256"],
    )

    config = load_campaign_config(_write_recipe(tmp_path, raw))

    assert config.experiment("full75-e8").tuning_authorities == (
        "recovery-stats",
        "layer-attribution",
    )
    assert config.experiment("worst8-e8p").tuning_authorities == (
        "recovery-stats",
        "alternate-recovered-attribution",
    )


@pytest.mark.parametrize(
    ("tuning_authorities", "message"),
    (
        (["layer-attribution"], "recovery-stats.*exactly once"),
        (
            ["recovery-stats", "recovery-stats", "layer-attribution"],
            "recovery-stats.*exactly once",
        ),
        (["recovery-stats"], "exactly one attribution"),
        (
            ["recovery-stats", "layer-attribution", "family-policy"],
            "exactly one attribution",
        ),
    ),
)
def test_transition_requires_one_stats_and_one_attribution_tuning_authority(
    tmp_path: Path,
    tuning_authorities: list[str],
    message: str,
) -> None:
    raw = _raw_recipe()
    raw["experiments"][0]["tuning_authorities"] = tuning_authorities

    with pytest.raises(CampaignConfigError, match=message):
        load_campaign_config(_write_recipe(tmp_path, raw))


@pytest.mark.parametrize(
    ("role", "split"),
    (
        ("artifact", None),
        ("tuning", "report"),
    ),
)
def test_transition_selected_attribution_must_be_selection_tuning_authority(
    tmp_path: Path,
    role: str,
    split: str | None,
) -> None:
    raw = _raw_recipe()
    attribution = _authority_raw(raw, "layer-attribution")
    attribution["role"] = role
    attribution["split"] = split

    with pytest.raises(CampaignConfigError, match="selection|tuning"):
        load_campaign_config(_write_recipe(tmp_path, raw))


@pytest.mark.parametrize(
    ("flag", "value"),
    (
        ("--attribution-json", "artifacts/quality/wrong-attribution.json"),
        ("--expected-attribution-sha256", "2" * 64),
    ),
)
def test_transition_argv_must_match_experiment_selected_attribution(
    tmp_path: Path,
    flag: str,
    value: str,
) -> None:
    raw = _raw_recipe()
    recovered_attribution = deepcopy(_authority_raw(raw, "layer-attribution"))
    recovered_attribution.update(
        {
            "name": "alternate-recovered-attribution",
            "path": "artifacts/quality/glm52-recovery-full75-attribution.json",
            "sha256": "1" * 64,
        }
    )
    raw["authorities"].append(recovered_attribution)
    raw["experiments"][1]["tuning_authorities"] = [
        "recovery-stats",
        "alternate-recovered-attribution",
    ]
    transition = _transition_raw(raw, "worst8-e8p-rematerialize")
    _replace_option(
        transition["argv"],
        "--attribution-json",
        recovered_attribution["path"],
    )
    _replace_option(
        transition["argv"],
        "--expected-attribution-sha256",
        recovered_attribution["sha256"],
    )
    _replace_option(transition["argv"], flag, value)

    with pytest.raises(
        CampaignConfigError,
        match=f"{flag}|alternate-recovered-attribution",
    ):
        load_campaign_config(_write_recipe(tmp_path, raw))


@pytest.mark.parametrize(
    "artifact_path",
    (
        "artifacts/quality/glm52-recovery-wave1-mixed75-artifact-20260710",
        "artifacts/quality/a-sibling-candidate/conversion-manifest.json",
    ),
)
def test_transition_artifacts_must_be_strict_output_descendants(
    tmp_path: Path, artifact_path: str
) -> None:
    raw = _raw_recipe()
    transition = _transition_raw(raw, "full75-rematerialize")
    transition["expected_artifacts"] = [artifact_path]

    with pytest.raises(CampaignConfigError, match="strict descendant"):
        load_campaign_config(_write_recipe(tmp_path, raw))


@pytest.mark.parametrize("artifacts", ([], ["wrong-descendant"]))
def test_rematerialize_requires_exact_conversion_manifest_artifact(
    tmp_path: Path, artifacts: list[str]
) -> None:
    raw = _raw_recipe()
    transition = _transition_raw(raw, "full75-rematerialize")
    if artifacts:
        output = raw["experiments"][0]["output_path"]
        transition["expected_artifacts"] = [f"{output}/{artifacts[0]}"]
    else:
        transition["expected_artifacts"] = []

    with pytest.raises(CampaignConfigError, match="conversion-manifest.json"):
        load_campaign_config(_write_recipe(tmp_path, raw))


@pytest.mark.parametrize("failure", ("duplicate", "missing-value"))
def test_required_transition_options_are_unambiguous(
    tmp_path: Path, failure: str
) -> None:
    raw = _raw_recipe()
    transition = _transition_raw(raw, "full75-rematerialize")
    argv = transition["argv"]
    index = argv.index("--output-dir")
    if failure == "duplicate":
        argv[index:index] = ["--output-dir", raw["experiments"][0]["output_path"]]
    else:
        del argv[index + 1]

    with pytest.raises(CampaignConfigError, match="duplicate|required value"):
        load_campaign_config(_write_recipe(tmp_path, raw))


@pytest.mark.parametrize("option", ("abbreviated", "unknown"))
def test_rematerialize_options_use_exact_allowlist(
    tmp_path: Path, option: str
) -> None:
    raw = _raw_recipe()
    transition = _transition_raw(raw, "full75-rematerialize")
    argv = transition["argv"]
    if option == "abbreviated":
        argv[argv.index("--output-dir")] = "--output"
    else:
        argv.extend(["--not-a-campaign-option", "value"])

    with pytest.raises(CampaignConfigError, match="unknown option"):
        load_campaign_config(_write_recipe(tmp_path, raw))


@pytest.mark.parametrize(
    ("flag", "value"),
    (
        ("--source-dir", "/tmp/wrong-source"),
        ("--index-path", "/tmp/wrong-index.json"),
        ("--expected-full-source-blob-inventory-sha256", "0" * 64),
        ("--expected-routed-source-blob-inventory-sha256", "0" * 64),
    ),
)
def test_rematerialize_options_bind_typed_model_source(
    tmp_path: Path, flag: str, value: str
) -> None:
    raw = _raw_recipe()
    transition = _transition_raw(raw, "full75-rematerialize")
    _replace_option(transition["argv"], flag, value)

    with pytest.raises(CampaignConfigError, match=flag):
        load_campaign_config(_write_recipe(tmp_path, raw))


def test_worst_layer_count_matches_recovered_layers(tmp_path: Path) -> None:
    raw = _raw_recipe()
    transition = _transition_raw(raw, "full75-rematerialize")
    _replace_option(transition["argv"], "--worst-layer-count", "74")

    with pytest.raises(CampaignConfigError, match="--worst-layer-count"):
        load_campaign_config(_write_recipe(tmp_path, raw))


@pytest.mark.parametrize("failure", ("missing", "wrong", "unexpected"))
def test_e8p_layer_count_is_present_iff_complete_e8p_layers_exist(
    tmp_path: Path, failure: str
) -> None:
    raw = _raw_recipe()
    if failure == "unexpected":
        transition = _transition_raw(raw, "full75-rematerialize")
        resume_index = transition["argv"].index("--resume")
        transition["argv"][resume_index:resume_index] = [
            "--e8p-worst-layer-count",
            "1",
        ]
    else:
        transition = _transition_raw(raw, "worst8-e8p-rematerialize")
        if failure == "missing":
            _remove_option(transition["argv"], "--e8p-worst-layer-count")
        else:
            _replace_option(transition["argv"], "--e8p-worst-layer-count", "7")

    with pytest.raises(CampaignConfigError, match="--e8p-worst-layer-count"):
        load_campaign_config(_write_recipe(tmp_path, raw))


def test_e8p_layer_identities_match_ordered_recovery_prefix(tmp_path: Path) -> None:
    raw = _raw_recipe()
    worst8 = raw["experiments"][1]
    worst8["recovered_layers"][0:2] = reversed(worst8["recovered_layers"][0:2])

    with pytest.raises(CampaignConfigError, match="ordered.*prefix"):
        load_campaign_config(_write_recipe(tmp_path, raw))


def test_executable_campaign_caps_e8p_layers_at_16(tmp_path: Path) -> None:
    raw = _raw_recipe()
    experiment = raw["experiments"][1]
    layers = raw["experiments"][0]["recovered_layers"][:17]
    experiment["recovered_layers"] = layers
    experiment["projection_rates"] = {
        str(layer): {"gate": 16, "up": 16, "down": 16} for layer in layers
    }
    increment = raw["budget"]["e8p_increment_per_projection_bytes"]
    experiment["expected_payload_bytes"] = (
        raw["budget"]["accepted_payload_bytes"] + len(layers) * 3 * increment
    )
    assert experiment["expected_payload_bytes"] < raw["budget"]["payload_limit_bytes"]
    transition = _transition_raw(raw, "worst8-e8p-rematerialize")
    _replace_option(transition["argv"], "--worst-layer-count", "17")
    _replace_option(transition["argv"], "--e8p-worst-layer-count", "17")

    with pytest.raises(CampaignConfigError, match="at most 16"):
        load_campaign_config(_write_recipe(tmp_path, raw))


def test_executable_rematerialize_requires_nonempty_recovered_layers(
    tmp_path: Path,
) -> None:
    raw = _raw_recipe()
    experiment = raw["experiments"][0]
    experiment["recovered_layers"] = []
    _replace_option(
        _transition_raw(raw, "full75-rematerialize")["argv"],
        "--worst-layer-count",
        "0",
    )

    with pytest.raises(CampaignConfigError, match="recovered_layers.*nonempty"):
        load_campaign_config(_write_recipe(tmp_path, raw))


def test_rematerialize_requires_resume(tmp_path: Path) -> None:
    raw = _raw_recipe()
    transition = _transition_raw(raw, "full75-rematerialize")
    _remove_option(transition["argv"], "--resume", has_value=False)

    with pytest.raises(CampaignConfigError, match="--resume"):
        load_campaign_config(_write_recipe(tmp_path, raw))


def test_transition_argv_rejects_shell_execution(tmp_path: Path) -> None:
    raw = _raw_recipe()
    transition = _transition_raw(raw, "full75-rematerialize")
    transition["argv"][:3] = ["/bin/sh", "-c", "true"]

    with pytest.raises(CampaignConfigError, match="shell"):
        load_campaign_config(_write_recipe(tmp_path, raw))


def test_transition_argv_rejects_wired_limit_override(tmp_path: Path) -> None:
    raw = _raw_recipe()
    transition = _transition_raw(raw, "full75-rematerialize")
    transition["argv"].append("GLM_MLX_WIRED_LIMIT_GB=128")

    with pytest.raises(CampaignConfigError, match="wired-memory limit"):
        load_campaign_config(_write_recipe(tmp_path, raw))


def test_transition_rejects_unsupported_action(tmp_path: Path) -> None:
    raw = _raw_recipe()
    transition = _transition_raw(raw, "full75-rematerialize")
    transition["action"] = "audit"

    with pytest.raises(CampaignConfigError, match="unsupported action"):
        load_campaign_config(_write_recipe(tmp_path, raw))


def test_orphan_transition_fails_closed_before_it_can_bypass_validation(
    tmp_path: Path,
) -> None:
    raw = _raw_recipe()
    orphan = deepcopy(_transition_raw(raw, "full75-rematerialize"))
    orphan["name"] = "orphan-transition"
    orphan["action"] = "audit"
    orphan["argv"][2] = "audit"
    orphan["argv"].extend(["--unsafe-orphan-option", "value"])
    raw["transitions"].append(orphan)

    with pytest.raises(CampaignConfigError, match="exactly one experiment"):
        load_campaign_config(_write_recipe(tmp_path, raw))


def test_transition_cannot_be_named_by_two_experiments(tmp_path: Path) -> None:
    raw = _raw_recipe()
    raw["experiments"][1]["transition"] = "full75-rematerialize"

    with pytest.raises(CampaignConfigError, match="exactly one experiment"):
        load_campaign_config(_write_recipe(tmp_path, raw))


def test_nested_duplicate_yaml_key_fails_closed(tmp_path: Path) -> None:
    text = RECIPE.read_text()
    duplicated = text.replace(
        "    heavy: true\n    argv:\n",
        "    heavy: true\n    heavy: false\n    argv:\n",
        1,
    )
    assert duplicated != text

    with pytest.raises(CampaignConfigError, match="duplicate key.*heavy"):
        load_campaign_config(_write_raw_recipe(tmp_path, duplicated))


@pytest.mark.parametrize("dependency", ("not-an-experiment", "worst8-e8p"))
def test_unknown_or_forward_dependencies_fail_closed(
    tmp_path: Path, dependency: str
) -> None:
    raw = _raw_recipe()
    raw["experiments"][0]["depends_on"] = [dependency]

    with pytest.raises(CampaignConfigError, match="unknown or forward"):
        load_campaign_config(_write_recipe(tmp_path, raw))


def test_unknown_authority_reference_fails_closed(tmp_path: Path) -> None:
    raw = _raw_recipe()
    raw["experiments"][0]["tuning_authorities"].append("not-an-authority")

    with pytest.raises(CampaignConfigError, match="unknown authority"):
        load_campaign_config(_write_recipe(tmp_path, raw))


def test_transition_target_mismatch_fails_closed(tmp_path: Path) -> None:
    raw = _raw_recipe()
    transition = _transition_raw(raw, "full75-rematerialize")
    transition["experiment"] = "worst8-e8p"

    with pytest.raises(CampaignConfigError, match="wrong experiment"):
        load_campaign_config(_write_recipe(tmp_path, raw))


def test_duplicate_authority_names_fail_closed(tmp_path: Path) -> None:
    raw = _raw_recipe()
    raw["authorities"][1]["name"] = raw["authorities"][0]["name"]

    with pytest.raises(CampaignConfigError, match="duplicate authority"):
        load_campaign_config(_write_recipe(tmp_path, raw))


def test_duplicate_transition_names_fail_closed(tmp_path: Path) -> None:
    raw = _raw_recipe()
    raw["transitions"][1]["name"] = raw["transitions"][0]["name"]

    with pytest.raises(CampaignConfigError, match="duplicate transition"):
        load_campaign_config(_write_recipe(tmp_path, raw))
