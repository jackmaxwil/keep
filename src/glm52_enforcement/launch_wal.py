"""Durable hash-chained SQLite journal for worker launch intent."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import threading
from typing import Mapping, Optional

from .canonical import canonical_json_bytes, canonical_sha256


class LaunchWalError(ValueError):
    """The prepared/committed launch journal is not coherent."""


class SqliteLaunchWal:
    """Append-only, full-synchronous journal with exact duplicate adoption."""

    def __init__(self, path: Path) -> None:
        if not isinstance(path, Path) or not path.is_absolute():
            raise LaunchWalError("launch WAL path must be absolute")
        self._path = path
        self._lock = threading.Lock()
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self._path), timeout=5.0)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS launch_journal (
                    sequence INTEGER PRIMARY KEY,
                    entry_kind TEXT NOT NULL,
                    operation_identity TEXT NOT NULL UNIQUE,
                    prior_entry_sha256 TEXT,
                    entry_body BLOB NOT NULL,
                    entry_sha256 TEXT NOT NULL UNIQUE
                )
                """
            )
        self._fsync()

    def _fsync(self) -> None:
        descriptor = os.open(str(self._path), os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        directory = os.open(str(self._path.parent), os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def _append(self, expected_kind: str, value: object) -> str:
        if type(value) is not dict or value.get("record_type") != expected_kind:
            raise LaunchWalError("launch journal entry kind is invalid")
        body = canonical_json_bytes(value)
        operation_identity = canonical_sha256(
            {
                "record_type": expected_kind,
                "activation_id": value.get("activation_id"),
                "allocation_ordinal": value.get("allocation_ordinal"),
                "prepared_journal_entry_sha256": value.get(
                    "prepared_journal_entry_sha256"
                ),
            }
        )
        with self._lock:
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                duplicate = connection.execute(
                    """
                    SELECT entry_body, entry_sha256
                    FROM launch_journal
                    WHERE operation_identity = ?
                    """,
                    (operation_identity,),
                ).fetchone()
                if duplicate is not None:
                    if duplicate[0] != body:
                        raise LaunchWalError(
                            "same launch journal operation has different bytes"
                        )
                    connection.rollback()
                    return str(duplicate[1])
                previous = connection.execute(
                    """
                    SELECT sequence, entry_sha256
                    FROM launch_journal
                    ORDER BY sequence DESC
                    LIMIT 1
                    """
                ).fetchone()
                sequence = 1 if previous is None else int(previous[0]) + 1
                prior = None if previous is None else str(previous[1])
                entry_sha = hashlib.sha256(
                    canonical_json_bytes(
                        {
                            "sequence": sequence,
                            "entry_kind": expected_kind,
                            "operation_identity": operation_identity,
                            "prior_entry_sha256": prior,
                            "entry_body_sha256": hashlib.sha256(body).hexdigest(),
                        }
                    )
                ).hexdigest()
                connection.execute(
                    """
                    INSERT INTO launch_journal (
                        sequence,
                        entry_kind,
                        operation_identity,
                        prior_entry_sha256,
                        entry_body,
                        entry_sha256
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        sequence,
                        expected_kind,
                        operation_identity,
                        prior,
                        body,
                        entry_sha,
                    ),
                )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.close()
            self._fsync()
        return entry_sha

    def append_prepared(self, entry: Mapping[str, object]) -> str:
        return self._append("WORKER_LAUNCH_PREPARED", entry)

    def append_committed(self, entry: Mapping[str, object]) -> str:
        return self._append("WORKER_LAUNCH_DDB_COMMITTED", entry)

    def read_entry(self, entry_sha256: str) -> Optional[Mapping[str, object]]:
        if type(entry_sha256) is not str or len(entry_sha256) != 64:
            raise LaunchWalError("launch journal identity is invalid")
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT entry_body
                FROM launch_journal
                WHERE entry_sha256 = ?
                """,
                (entry_sha256,),
            ).fetchone()
        if row is None:
            return None
        value = json.loads(bytes(row[0]).decode("utf-8"))
        if type(value) is not dict:
            raise LaunchWalError("launch journal body is malformed")
        return value

    def read_prepared(
        self,
        activation_id: str,
        allocation_ordinal: int,
    ) -> Optional[Mapping[str, object]]:
        if (
            type(activation_id) is not str
            or not activation_id
            or type(allocation_ordinal) is not int
            or allocation_ordinal <= 0
        ):
            raise LaunchWalError("prepared launch coordinates are invalid")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT entry_body
                FROM launch_journal
                WHERE entry_kind = 'WORKER_LAUNCH_PREPARED'
                ORDER BY sequence ASC
                """
            ).fetchall()
        matches = []
        for row in rows:
            value = json.loads(bytes(row[0]).decode("utf-8"))
            if (
                type(value) is dict
                and value.get("activation_id") == activation_id
                and value.get("allocation_ordinal") == allocation_ordinal
            ):
                matches.append(value)
        if len(matches) > 1:
            raise LaunchWalError("multiple prepared WAL entries exist")
        return None if not matches else matches[0]

    def verify_chain(self) -> int:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT sequence, entry_kind, operation_identity,
                       prior_entry_sha256, entry_body, entry_sha256
                FROM launch_journal
                ORDER BY sequence ASC
                """
            ).fetchall()
        prior = None
        for expected_sequence, row in enumerate(rows, start=1):
            sequence, kind, operation, recorded_prior, body, recorded_sha = row
            if sequence != expected_sequence or recorded_prior != prior:
                raise LaunchWalError("launch journal sequence or chain drifted")
            expected_sha = hashlib.sha256(
                canonical_json_bytes(
                    {
                        "sequence": sequence,
                        "entry_kind": kind,
                        "operation_identity": operation,
                        "prior_entry_sha256": prior,
                        "entry_body_sha256": hashlib.sha256(body).hexdigest(),
                    }
                )
            ).hexdigest()
            if recorded_sha != expected_sha:
                raise LaunchWalError("launch journal entry hash drifted")
            prior = recorded_sha
        return len(rows)


__all__ = ["LaunchWalError", "SqliteLaunchWal"]
