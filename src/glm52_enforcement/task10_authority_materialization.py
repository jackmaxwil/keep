"""Conditional presend materialization for Task 10 retained authority."""

from __future__ import annotations

from dataclasses import asdict
import json
import re
from typing import Mapping

from .canonical import canonical_json_bytes
from .dynamodb import decode_item, encode_item
from .task10_sole_sender import (
    RUN_ID,
    Task10SoleSenderAuthority,
    validate_task10_sole_sender_authority,
)


_TABLE = re.compile(r"^[A-Za-z0-9_.-]{3,255}$")


class Task10AuthorityMaterializationError(ValueError):
    """The exact presend authority row could not be proven durable."""


def task10_sole_sender_authority_sort_key(
    value: Task10SoleSenderAuthority,
) -> str:
    value = validate_task10_sole_sender_authority(value)
    return (
        f"ACTIVATION#{value.activation_id}#"
        f"TASK10_SOLE_SENDER_AUTHORITY#{value.generation_text}"
    )


def task10_sole_sender_authority_item(
    value: Task10SoleSenderAuthority,
) -> dict[str, object]:
    value = validate_task10_sole_sender_authority(value)
    body = json.loads(canonical_json_bytes(asdict(value)))
    return {
        "PK": RUN_ID,
        "SK": task10_sole_sender_authority_sort_key(value),
        **body,
    }


def _metadata(response: object, label: str) -> None:
    if (
        type(response) is not dict
        or type(response.get("ResponseMetadata")) is not dict
        or response["ResponseMetadata"].get("HTTPStatusCode") != 200
        or type(response["ResponseMetadata"].get("RequestId")) is not str
        or not response["ResponseMetadata"]["RequestId"]
        or type(response["ResponseMetadata"].get("RetryAttempts")) is not int
        or response["ResponseMetadata"]["RetryAttempts"] != 0
    ):
        raise Task10AuthorityMaterializationError(
            label + " response is unauthenticated"
        )


def _readback(
    *,
    dynamodb: object,
    table_name: str,
    expected: Mapping[str, object],
) -> None:
    response = dynamodb.get_item(
        TableName=table_name,
        Key=encode_item({"PK": expected["PK"], "SK": expected["SK"]}),
        ConsistentRead=True,
        ReturnConsumedCapacity="NONE",
    )
    _metadata(response, "Task 10 authority GetItem")
    if type(response.get("Item")) is not dict:
        raise Task10AuthorityMaterializationError(
            "Task 10 authority is absent after write"
        )
    if decode_item(response["Item"]) != expected:
        raise Task10AuthorityMaterializationError(
            "foreign Task 10 authority already exists"
        )


def materialize_task10_sole_sender_authority(
    value: Task10SoleSenderAuthority,
    *,
    dynamodb: object,
    table_name: str,
) -> Mapping[str, object]:
    """Write once, then reconcile both conditional and lost responses."""

    if type(table_name) is not str or _TABLE.fullmatch(table_name) is None:
        raise Task10AuthorityMaterializationError(
            "Task 10 authority table name drifted"
        )
    expected = task10_sole_sender_authority_item(value)
    outcome = "WRITTEN"
    try:
        response = dynamodb.put_item(
            TableName=table_name,
            Item=encode_item(expected),
            ConditionExpression=(
                "attribute_not_exists(PK) AND attribute_not_exists(SK)"
            ),
            ReturnConsumedCapacity="NONE",
        )
        _metadata(response, "Task 10 authority PutItem")
    except Exception:
        outcome = "RECONCILED"
    _readback(
        dynamodb=dynamodb,
        table_name=table_name,
        expected=expected,
    )
    return {
        "outcome": outcome,
        "partition_key": expected["PK"],
        "sort_key": expected["SK"],
        "authority_identity_sha256": value.canonical_identity_sha256,
    }


__all__ = [
    "Task10AuthorityMaterializationError",
    "materialize_task10_sole_sender_authority",
    "task10_sole_sender_authority_item",
    "task10_sole_sender_authority_sort_key",
]
