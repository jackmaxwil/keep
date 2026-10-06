"""Task 13 semantic transport-gate evidence and anti-forgery contracts."""

from __future__ import annotations

import base64
import copy
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path

import pytest

from glm52_enforcement.canonical import canonical_json_bytes

REPO_ROOT = Path(__file__).parents[1]
BUILDER = REPO_ROOT / "aws/glm52-gpu/scripts/build_glm52_task13_transport_gates.py"
PREQUALIFICATION_BUILDER = (
    REPO_ROOT / "aws/glm52-gpu/scripts/build_glm52_task13_prequalification_inputs.py"
)
LAUNCH_BRIDGE_BUILDER = (
    REPO_ROOT / "aws/glm52-gpu/scripts/build_glm52_task13_launch_bridge.py"
)
GATE_PUBLISHER = (
    REPO_ROOT / "aws/glm52-gpu/scripts/publish_glm52_task13_transport_gates.py"
)


def _api():
    try:
        return importlib.import_module("glm52_enforcement.task13_transport_gates")
    except ModuleNotFoundError:
        pytest.fail(
            "Task 13 semantic transport-gate implementation is missing",
            pytrace=False,
        )


def test_red_exact_verbatim_transport_matrix_and_obligation_inventory_exists() -> None:
    """Break caught: Task 13 substitutes gate labels for the frozen obligations."""

    api = _api()

    assert tuple(row.row_id for row in api.T01_T25_ROWS) == tuple(
        f"T{index:02d}" for index in range(1, 26)
    )
    assert tuple(
        row.obligation_id for row in api.TRANSPORT_MUTANT_OBLIGATIONS
    ) == tuple(f"M{index:02d}" for index in range(1, 23))
    assert api.T01_T25_ROWS[0].operation_result == (
        "direct unambiguous claim create `200` with exact owner"
    )
    assert api.T01_T25_ROWS[-1].durable_interpretation == (
        "durable reconcile-only decision"
    )
    assert api.TRANSPORT_MUTANT_OBLIGATIONS[0].text.startswith(
        "claim, decision, and terminal `409`, `412`, timeout"
    )
    assert api.TRANSPORT_MUTANT_OBLIGATIONS[-1].text.endswith(
        "cached head state or an inactive selected child authorizes none of them."
    )


def _observed(nodeid: str):
    api = _api()
    return api.PytestObservation(
        nodeid=nodeid,
        exit_code=0,
        passed_count=1,
        normalized_output_sha256=hashlib.sha256(
            ("observed:" + nodeid).encode()
        ).hexdigest(),
    )


def _rehash(value: dict[str, object]) -> None:
    unhashed = dict(value)
    unhashed.pop("canonical_identity_sha256", None)
    value["canonical_identity_sha256"] = hashlib.sha256(
        canonical_json_bytes(unhashed)
    ).hexdigest()


def test_red_gate_builders_bind_exact_routes_and_observed_pytest_outcomes() -> None:
    """Break caught: coordinate labels or caller booleans impersonate evidence."""

    api = _api()
    t_gate = api.build_t01_t25_gate(
        repo_root=REPO_ROOT,
        observe=_observed,
    )
    mutant_gate = api.build_transport_mutant_gate(
        repo_root=REPO_ROOT,
        observe=_observed,
    )

    assert t_gate["row_count"] == 25
    assert t_gate["status"] == "PROVEN"
    assert [row["row_id"] for row in t_gate["rows"]] == [
        f"T{index:02d}" for index in range(1, 26)
    ]
    assert mutant_gate["obligation_count"] == 22
    assert mutant_gate["killed_count"] == 22
    assert mutant_gate["survived_count"] == 0
    assert mutant_gate["masked_count"] == 0
    assert [row["obligation_id"] for row in mutant_gate["obligations"]] == [
        f"M{index:02d}" for index in range(1, 23)
    ]

    assert (
        api.validate_t01_t25_gate(
            t_gate,
            repo_root=REPO_ROOT,
            observe=_observed,
        )
        == t_gate
    )
    assert (
        api.validate_transport_mutant_gate(
            mutant_gate,
            repo_root=REPO_ROOT,
            observe=_observed,
        )
        == mutant_gate
    )

    for receipt in [*t_gate["rows"], *mutant_gate["obligations"]]:
        assert receipt["pytest_nodeids"]
        assert receipt["route_paths"]
        assert receipt["files_sha256"]
        assert receipt["observations"]
        assert all(
            observation["exit_code"] == 0 for observation in receipt["observations"]
        )


def test_red_strict_validator_rejects_omission_duplication_order_and_no_observer() -> (
    None
):
    """Break caught: a partial or reordered artifact is accepted by its rehash."""

    api = _api()
    gate = api.build_t01_t25_gate(
        repo_root=REPO_ROOT,
        observe=_observed,
    )
    with pytest.raises(api.TransportGateError, match="observer"):
        api.validate_t01_t25_gate(gate, repo_root=REPO_ROOT)

    variants = []
    omitted = copy.deepcopy(gate)
    omitted["rows"].pop()
    omitted["row_count"] = 24
    _rehash(omitted)
    variants.append(omitted)

    duplicated = copy.deepcopy(gate)
    duplicated["rows"][-1] = copy.deepcopy(duplicated["rows"][0])
    _rehash(duplicated)
    variants.append(duplicated)

    reordered = copy.deepcopy(gate)
    reordered["rows"][0], reordered["rows"][1] = (
        reordered["rows"][1],
        reordered["rows"][0],
    )
    _rehash(reordered)
    variants.append(reordered)

    for variant in variants:
        with pytest.raises(api.TransportGateError):
            api.validate_t01_t25_gate(
                variant,
                repo_root=REPO_ROOT,
                observe=_observed,
            )


def test_red_forged_rehash_cannot_replace_a_fresh_observation() -> None:
    """Break caught: changing a receipt to PASS and rehashing becomes proof."""

    api = _api()
    gate = api.build_transport_mutant_gate(
        repo_root=REPO_ROOT,
        observe=_observed,
    )
    forged = copy.deepcopy(gate)
    forged_observation = forged["obligations"][0]["observations"][0]
    forged_observation["normalized_output_sha256"] = "f" * 64
    _rehash(forged)

    with pytest.raises(api.TransportGateError, match="observation"):
        api.validate_transport_mutant_gate(
            forged,
            repo_root=REPO_ROOT,
            observe=_observed,
        )


def test_red_pytest_summary_parser_accepts_the_observed_newline_boundary() -> None:
    """Break caught: a real passing selector is rejected as unobserved."""

    api = _api()
    assert (
        api._passing_pytest_count(
            ".                                                                        [100%]\n"
            "1 passed in 0.07s\n"
        )
        == 1
    )


def test_red_builder_is_fixed_to_both_canonical_gate_names() -> None:
    """Break caught: one gate can be generated or replaced independently."""

    assert BUILDER.is_file()
    source = BUILDER.read_text()
    assert "t01-t25.json" in source
    assert "transport-22-mutants.json" in source
    assert "validate_t01_t25_gate" in source
    assert "validate_transport_mutant_gate" in source
    assert "os.O_EXCL" in source


def test_red_package_coordinate_pins_bind_the_observed_semantic_artifacts() -> None:
    """Break caught: any well-shaped coordinate can claim transport proof."""

    api = _api()
    for artifact_kind, pin in api.SEMANTIC_GATE_PINS.items():
        assert (
            api.validate_semantic_gate_coordinate(
                {
                    "artifact_kind": artifact_kind,
                    "file_sha256": pin["file_sha256"],
                    "body_sha256": pin["body_sha256"],
                }
            )
            == pin
        )

        forged = {
            "artifact_kind": artifact_kind,
            "file_sha256": pin["file_sha256"],
            "body_sha256": "f" * 64,
        }
        with pytest.raises(api.TransportGateError, match="semantic"):
            api.validate_semantic_gate_coordinate(forged)


def test_checked_gate_files_are_canonical_and_equal_the_package_pins() -> None:
    """Break caught: checked evidence bytes drift from package authority."""

    api = _api()
    paths = {
        "T01_T25_GATE": REPO_ROOT / "task13/gates/t01-t25.json",
        "TRANSPORT_22_MUTANT_GATE": (
            REPO_ROOT / "task13/gates/transport-22-mutants.json"
        ),
    }
    for artifact_kind, path in paths.items():
        raw = path.read_bytes()
        value = json.loads(raw)
        assert raw == canonical_json_bytes(value) + b"\n"
        assert (
            hashlib.sha256(raw).hexdigest()
            == (api.SEMANTIC_GATE_PINS[artifact_kind]["file_sha256"])
        )
        assert (
            value["canonical_identity_sha256"]
            == (api.SEMANTIC_GATE_PINS[artifact_kind]["body_sha256"])
        )


def test_red_task13_has_owned_prequalification_and_launch_bridge_builders() -> None:
    """Break caught: Task 13 needs hand-authored launch boundary files."""

    assert PREQUALIFICATION_BUILDER.is_file()
    assert LAUNCH_BRIDGE_BUILDER.is_file()
    prequalification_source = PREQUALIFICATION_BUILDER.read_text()
    bridge_source = LAUNCH_BRIDGE_BUILDER.read_text()
    assert "--base-request" in prequalification_source
    assert "--reviewed-artifacts" in prequalification_source
    assert "build_campaign_package" in prequalification_source
    assert "SEMANTIC_GATE_PINS" in prequalification_source
    assert "task13_launch_bridge_from_mapping" in bridge_source
    assert "canonical_identity_sha256" in bridge_source
    assert "O_EXCL" in bridge_source
    assert GATE_PUBLISHER.is_file()
    publisher_source = GATE_PUBLISHER.read_text()
    assert 'IfNoneMatch="*"' in publisher_source
    assert "ExpectedBucketOwner=ACCOUNT_ID" in publisher_source
    assert "validate_t01_t25_gate" in publisher_source
    assert "validate_transport_mutant_gate" in publisher_source
    assert "list_object_versions" in publisher_source


@pytest.mark.parametrize("ambiguous", [False, True])
def test_gate_publisher_is_one_conditional_put_then_exact_reconciliation(
    ambiguous: bool,
) -> None:
    spec = importlib.util.spec_from_file_location(
        "task13_gate_publisher_test",
        GATE_PUBLISHER,
    )
    assert spec is not None and spec.loader is not None
    publisher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publisher)
    raw = (REPO_ROOT / "task13/gates/t01-t25.json").read_bytes()
    checksum = base64.b64encode(hashlib.sha256(raw).digest()).decode("ascii")

    class Body:
        def read(self):
            return raw

        def close(self):
            return None

    class Client:
        def __init__(self):
            self.puts = []

        def put_object(self, **kwargs):
            self.puts.append(kwargs)
            if ambiguous:
                raise TimeoutError("response lost")
            return {"VersionId": "version-1"}

        def list_object_versions(self, **_kwargs):
            return {
                "IsTruncated": False,
                "Versions": [
                    {
                        "Key": "task13/gates/t01-t25.json",
                        "VersionId": "version-1",
                    }
                ],
                "DeleteMarkers": [],
            }

        def get_object(self, **_kwargs):
            return {
                "VersionId": "version-1",
                "ChecksumSHA256": checksum,
                "Body": Body(),
            }

    client = Client()
    coordinate = publisher._publish_one(
        client,
        artifact_kind="T01_T25_GATE",
        key="task13/gates/t01-t25.json",
        raw=raw,
        reconcilable_errors=(TimeoutError,),
    )
    assert len(client.puts) == 1
    assert client.puts[0]["IfNoneMatch"] == "*"
    assert client.puts[0]["ExpectedBucketOwner"] == "246813579024"
    assert coordinate["version_id"] == "version-1"
