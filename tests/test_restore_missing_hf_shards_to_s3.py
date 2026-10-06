from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "aws/glm52-gpu/scripts/restore_missing_hf_shards_to_s3.py"
try:
    import boto3 as _boto3  # noqa: F401
except ModuleNotFoundError:
    stubbed_boto3 = True
    sys.modules["boto3"] = types.ModuleType("boto3")
else:
    stubbed_boto3 = False
try:
    from botocore.exceptions import ClientError as _ClientError  # noqa: F401
except ModuleNotFoundError:
    stubbed_botocore = True
    botocore = types.ModuleType("botocore")
    botocore_exceptions = types.ModuleType("botocore.exceptions")

    class ClientError(Exception):
        pass

    botocore_exceptions.ClientError = ClientError
    sys.modules["botocore"] = botocore
    sys.modules["botocore.exceptions"] = botocore_exceptions
else:
    stubbed_botocore = False
SPEC = importlib.util.spec_from_file_location("_glm52_source_restore", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RESTORE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RESTORE
SPEC.loader.exec_module(RESTORE)
if stubbed_botocore:
    sys.modules.pop("botocore.exceptions", None)
    sys.modules.pop("botocore", None)
if stubbed_boto3:
    sys.modules.pop("boto3", None)


def test_multipart_buffer_never_emits_an_undersized_nonfinal_part() -> None:
    mib = 1024 * 1024
    pending = bytearray()

    pending.extend(b"a" * (3 * mib))
    assert RESTORE._drain_multipart_parts(  # type: ignore[attr-defined]
        pending,
        part_bytes=5 * mib,
        final=False,
    ) == []

    pending.extend(b"b" * (3 * mib))
    full_parts = RESTORE._drain_multipart_parts(  # type: ignore[attr-defined]
        pending,
        part_bytes=5 * mib,
        final=False,
    )
    assert full_parts == [b"a" * (3 * mib) + b"b" * (2 * mib)]
    assert pending == b"b" * mib

    pending.extend(b"c" * mib)
    final_parts = RESTORE._drain_multipart_parts(  # type: ignore[attr-defined]
        pending,
        part_bytes=5 * mib,
        final=True,
    )
    assert final_parts == [b"b" * mib + b"c" * mib]
    assert pending == b""
