from __future__ import annotations

import pytest


def _graph() -> dict[str, dict[str, object]]:
    from glm52_enforcement.task12_operations import (
        build_operation_descriptor_graph,
    )

    return build_operation_descriptor_graph(
        activation_id="activation-1",
        activation_ordinal=1,
        generation=1,
        ledger_partition_key="RUN#glm52-sky-20260724",
        campaign_bucket="glm52-sky-campaign",
        kms_key_id=(
            "arn:aws:kms:us-west-2:246813579024:key/"
            "11111111-2222-3333-4444-555555555555"
        ),
        static_authority_sha256="a" * 64,
    )


def test_descriptor_graph_has_32_closed_producer_bound_operations() -> None:
    from glm52_enforcement.task12_operations import (
        OPERATION_SPECS,
        validate_operation_descriptor,
        validate_operation_source_producers,
    )

    descriptors = _graph()

    assert set(descriptors) == set(OPERATION_SPECS)
    validate_operation_source_producers()
    assert all(row["source_coordinates"]["sources"] for row in descriptors.values())
    assert all(
        "live_source" not in {
            source["alias"] for source in row["source_coordinates"]["sources"]
        }
        for row in descriptors.values()
    )
    source_kinds = {
        source["coordinate_kind"]
        for row in descriptors.values()
        for source in row["source_coordinates"]["sources"]
    }
    assert source_kinds == {
        "DDB_EXACT",
        "DDB_DERIVED",
        "DDB_KEY_FROM_CONTROL",
        "S3_VERSIONED_FROM_CONTROL",
    }
    terminal_source = next(
        source
        for source in descriptors["RETAINED_ENTER_RECOVERY_COMPLETE"][
            "source_coordinates"
        ]["sources"]
        if source["alias"] == "terminal_v2"
    )
    assert terminal_source == {
        "alias": "terminal_v2",
        "coordinate_kind": "S3_VERSIONED_FROM_CONTROL",
        "record_type": "glm52_production_terminal_v2",
        "writer_kind": "TerminalV2",
        "campaign_bucket": "glm52-sky-campaign",
        "control": {
            "coordinate_kind": "DDB_EXACT",
            "record_type": "glm52_task12_versioned_writer_control_v1",
            "sort_key": (
                "ACTIVATION#activation-1#TASK12_VERSIONED_WRITER_CONTROL#"
                "00000001#TERMINALV2"
            ),
        },
    }
    assert all(
        validate_operation_descriptor(row) == row
        for row in descriptors.values()
    )
    for row in descriptors.values():
        source_kinds = {
            source["coordinate_kind"]
            for source in row["source_coordinates"]["sources"]
        }
        assert "dynamodb:GetItem" in row["allowed_read_apis"]
        if "S3_VERSIONED_FROM_CONTROL" in source_kinds:
            assert "s3:GetObjectVersion" in row["allowed_read_apis"]


def test_descriptor_rejects_cross_operation_source_reuse() -> None:
    from glm52_enforcement.task12_operations import (
        Task12OperationContractError,
        validate_operation_descriptor,
    )

    descriptors = _graph()
    first = dict(descriptors["RECONCILE_RETAINED_LIFECYCLE_TRIGGER"])
    first["source_coordinates"] = dict(first["source_coordinates"])
    first["source_coordinates"]["sources"] = [{"alias": "foreign"}]

    with pytest.raises(Task12OperationContractError):
        validate_operation_descriptor(first)


def test_versioned_s3_descriptor_rejects_missing_target_record_type() -> None:
    from glm52_enforcement.canonical import canonical_sha256
    from glm52_enforcement.task12_operations import (
        Task12OperationContractError,
        validate_operation_descriptor,
    )

    row = dict(_graph()["RETAINED_ENTER_RECOVERY_COMPLETE"])
    coordinates = dict(row["source_coordinates"])
    sources = [dict(source) for source in coordinates["sources"]]
    terminal = next(
        source for source in sources if source["alias"] == "terminal_v2"
    )
    terminal.pop("record_type")
    coordinates["sources"] = sources
    row["source_coordinates"] = coordinates
    row["canonical_body_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in row.items()
            if key != "canonical_body_sha256"
        }
    )

    with pytest.raises(Task12OperationContractError):
        validate_operation_descriptor(row)


def test_every_non_root_operation_consumes_a_real_prior_canonical_source() -> None:
    from glm52_enforcement.task12_operations import (
        EXTERNAL_PRODUCER,
        OPERATION_PREDECESSORS,
        OPERATION_PRODUCED_SOURCES,
        OPERATION_SOURCE_PRODUCERS,
    )

    for operation, predecessor in OPERATION_PREDECESSORS.items():
        if predecessor is None:
            continue
        produced_bindings = [
            (alias, producer)
            for alias, producer in OPERATION_SOURCE_PRODUCERS[operation]
            if producer != EXTERNAL_PRODUCER
        ]
        assert produced_bindings, operation
        assert all(
            alias in OPERATION_PRODUCED_SOURCES[producer]
            for alias, producer in produced_bindings
        )
