from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

from glm52_enforcement.canonical import canonical_json_bytes
from glm52_enforcement.task13_campaign_runner import FinalizationCapture
from glm52_enforcement.task13_live_inputs import Task13LiveInputError

ROOT = Path(__file__).resolve().parents[1]
PREQUALIFICATION_SCRIPT = (
    ROOT
    / "aws/glm52-gpu/scripts/build_glm52_task13_prequalification_sources.py"
)
MATERIALIZATION_SCRIPT = (
    ROOT
    / "aws/glm52-gpu/scripts/build_glm52_task10_materialization_manifest.py"
)
RUNNER_SCRIPT = (
    ROOT / "aws/glm52-gpu/scripts/run_glm52_task13_campaign.py"
)


def _load_cli(path: Path, name: str) -> object:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _direct_case(
    tmp_path: Path,
    script: Path,
    case: str,
) -> tuple[object, list[Path], list[bytes]]:
    module = _load_cli(script, "_launch_io_" + case)
    count = 3 if case == "prequalification" else 1
    paths = [
        (tmp_path / f"{case}-{index}.json").resolve()
        for index in range(count)
    ]
    values = [
        {"case": case, "index": index, "payload": "x" * 37}
        for index in range(count)
    ]
    expected = [canonical_json_bytes(value) + b"\n" for value in values]

    def invoke() -> None:
        if case == "prequalification":
            module._write_all(list(zip(paths, values)))
        else:
            module._write_new(paths[0], values[0])

    module._test_invoke = invoke
    return module, paths, expected


@pytest.mark.parametrize(
    ("script", "case"),
    [
        (PREQUALIFICATION_SCRIPT, "prequalification"),
        (MATERIALIZATION_SCRIPT, "materialization"),
    ],
)
def test_direct_writers_complete_short_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    script: Path,
    case: str,
) -> None:
    module, paths, expected = _direct_case(tmp_path, script, case)
    original_write = os.write

    def short_write(descriptor: int, remaining: memoryview) -> int:
        return original_write(descriptor, remaining[: min(3, len(remaining))])

    with monkeypatch.context() as patch:
        patch.setattr(module.os, "write", short_write)
        module._test_invoke()

    assert [path.read_bytes() for path in paths] == expected


@pytest.mark.parametrize(
    ("script", "case"),
    [
        (PREQUALIFICATION_SCRIPT, "prequalification"),
        (MATERIALIZATION_SCRIPT, "materialization"),
    ],
)
def test_direct_writers_reject_zero_progress_cleanup_and_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    script: Path,
    case: str,
) -> None:
    module, paths, expected = _direct_case(tmp_path, script, case)

    with monkeypatch.context() as patch:
        patch.setattr(module.os, "write", lambda _descriptor, _raw: 0)
        with pytest.raises(OSError, match="no progress"):
            module._test_invoke()

    assert all(not path.exists() for path in paths)
    module._test_invoke()
    assert [path.read_bytes() for path in paths] == expected


@pytest.mark.parametrize(
    ("script", "case", "fault"),
    [
        (PREQUALIFICATION_SCRIPT, "prequalification", "write"),
        (PREQUALIFICATION_SCRIPT, "prequalification", "fsync"),
        (MATERIALIZATION_SCRIPT, "materialization", "write"),
        (MATERIALIZATION_SCRIPT, "materialization", "fsync"),
    ],
)
def test_direct_writer_primary_io_failure_survives_close_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    script: Path,
    case: str,
    fault: str,
) -> None:
    module, paths, expected = _direct_case(tmp_path, script, case)
    original_close = os.close
    close_failed = False

    def close_then_fail(descriptor: int) -> None:
        nonlocal close_failed
        original_close(descriptor)
        if not close_failed:
            close_failed = True
            raise OSError("synthetic cleanup close failure")

    def primary_write_failure(_descriptor: int, _raw: memoryview) -> int:
        raise OSError("synthetic primary write failure")

    def primary_fsync_failure(_descriptor: int) -> None:
        raise OSError("synthetic primary fsync failure")

    with monkeypatch.context() as patch:
        patch.setattr(module.os, "close", close_then_fail)
        if fault == "write":
            patch.setattr(module.os, "write", primary_write_failure)
        else:
            patch.setattr(module.os, "fsync", primary_fsync_failure)
        with pytest.raises(OSError, match=f"primary {fault} failure"):
            module._test_invoke()

    assert close_failed
    assert all(not path.exists() for path in paths)
    module._test_invoke()
    assert [path.read_bytes() for path in paths] == expected


@pytest.mark.parametrize(
    ("script", "case"),
    [
        (PREQUALIFICATION_SCRIPT, "prequalification"),
        (MATERIALIZATION_SCRIPT, "materialization"),
    ],
)
def test_direct_writers_reject_late_close_failure_without_poisoning_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    script: Path,
    case: str,
) -> None:
    module, paths, expected = _direct_case(tmp_path, script, case)
    original_close = os.close
    close_failed = False
    close_calls = 0

    def close_then_fail(descriptor: int) -> None:
        nonlocal close_calls, close_failed
        close_calls += 1
        original_close(descriptor)
        if not close_failed:
            close_failed = True
            raise OSError("synthetic close failure")

    with monkeypatch.context() as patch:
        patch.setattr(module.os, "close", close_then_fail)
        with pytest.raises(OSError, match="synthetic close failure"):
            module._test_invoke()

    assert close_failed
    assert close_calls == len(paths)
    assert all(not path.exists() for path in paths)
    module._test_invoke()
    assert [path.read_bytes() for path in paths] == expected


@pytest.mark.parametrize(
    ("script", "case"),
    [
        (PREQUALIFICATION_SCRIPT, "prequalification"),
        (MATERIALIZATION_SCRIPT, "materialization"),
    ],
)
def test_direct_writers_preserve_preexisting_output(
    tmp_path: Path,
    script: Path,
    case: str,
) -> None:
    module, paths, _expected = _direct_case(tmp_path, script, case)
    collision = paths[-1]
    collision.write_bytes(b"foreign\n")

    with pytest.raises(Task13LiveInputError, match="output"):
        module._test_invoke()

    assert collision.read_bytes() == b"foreign\n"
    assert all(not path.exists() for path in paths[:-1])


def _capture() -> FinalizationCapture:
    return FinalizationCapture(
        status_code=200,
        executed_version="19",
        payload_bytes=b'{"payload":"' + b"p" * 43 + b'"}',
        gate_bytes=b'{"gate":"' + b"g" * 41 + b'"}',
    )


def _capture_outputs(module: object, tmp_path: Path) -> tuple[object, list[Path]]:
    paths = [
        (tmp_path / "metadata.json").resolve(),
        (tmp_path / "payload.json").resolve(),
        (tmp_path / "gate.json").resolve(),
    ]
    outputs = module._FinalizationCaptureFiles(
        metadata_path=paths[0],
        payload_path=paths[1],
        gate_path=paths[2],
    )
    return outputs, paths


def test_finalization_capture_completes_short_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_cli(RUNNER_SCRIPT, "_capture_short_write_io")
    outputs, paths = _capture_outputs(module, tmp_path)
    original_write = os.write

    def short_write(descriptor: int, remaining: memoryview) -> int:
        return original_write(
            descriptor,
            remaining[: min(2, len(remaining))],
        )

    with monkeypatch.context() as patch:
        patch.setattr(module.os, "write", short_write)
        outputs(_capture())
        outputs.commit()

    assert paths[0].read_bytes() == canonical_json_bytes(
        {"StatusCode": 200, "ExecutedVersion": "19"}
    ) + b"\n"
    assert paths[1].read_bytes() == _capture().payload_bytes
    assert paths[2].read_bytes() == _capture().gate_bytes


@pytest.mark.parametrize("fault", ["zero", "write", "fsync"])
def test_finalization_capture_io_failure_cleans_outputs_and_allows_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    module = _load_cli(RUNNER_SCRIPT, "_capture_" + fault + "_io")
    outputs, paths = _capture_outputs(module, tmp_path)

    with monkeypatch.context() as patch:
        if fault == "zero":
            patch.setattr(module.os, "write", lambda _descriptor, _raw: 0)
            match = "no progress"
        elif fault == "write":

            def fail_write(_descriptor: int, _raw: memoryview) -> int:
                raise OSError("synthetic capture write failure")
            patch.setattr(module.os, "write", fail_write)
            match = "capture write failure"
        else:

            def fail_fsync(_descriptor: int) -> None:
                raise OSError("synthetic capture fsync failure")
            patch.setattr(module.os, "fsync", fail_fsync)
            match = "capture fsync failure"
        with pytest.raises(OSError, match=match):
            outputs(_capture())

    assert all(not path.exists() for path in paths)
    retry, _retry_paths = _capture_outputs(module, tmp_path)
    retry(_capture())
    retry.commit()
    assert all(path.is_file() for path in paths)


def test_finalization_capture_preserves_primary_failure_during_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_cli(RUNNER_SCRIPT, "_capture_primary_failure_io")
    outputs, paths = _capture_outputs(module, tmp_path)
    original_close = os.close
    original_unlink = os.unlink

    def fail_write(_descriptor: int, _raw: memoryview) -> int:
        raise OSError("synthetic primary capture write failure")

    def close_then_fail(descriptor: int) -> None:
        original_close(descriptor)
        raise OSError("synthetic cleanup close failure")

    def unlink_then_fail(path: Path) -> None:
        original_unlink(path)
        raise OSError("synthetic cleanup unlink failure")

    with monkeypatch.context() as patch:
        patch.setattr(module.os, "write", fail_write)
        patch.setattr(module.os, "close", close_then_fail)
        patch.setattr(module.os, "unlink", unlink_then_fail)
        with pytest.raises(OSError, match="primary capture write failure"):
            outputs(_capture())

    assert all(not path.exists() for path in paths)


def test_finalization_capture_close_failure_aborts_and_allows_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_cli(RUNNER_SCRIPT, "_capture_close_io")
    outputs, paths = _capture_outputs(module, tmp_path)
    outputs(_capture())
    original_close = os.close
    close_failed = False
    close_calls = 0

    def close_then_fail(descriptor: int) -> None:
        nonlocal close_calls, close_failed
        close_calls += 1
        original_close(descriptor)
        if not close_failed:
            close_failed = True
            raise OSError("synthetic finalization close failure")

    with monkeypatch.context() as patch:
        patch.setattr(module.os, "close", close_then_fail)
        with pytest.raises(OSError, match="finalization close failure"):
            outputs.commit()

    assert close_failed
    assert close_calls == len(paths)
    assert all(not path.exists() for path in paths)
    retry, _retry_paths = _capture_outputs(module, tmp_path)
    retry(_capture())
    retry.commit()
    assert all(path.is_file() for path in paths)


def test_finalization_capture_collision_preserves_preexisting_file(
    tmp_path: Path,
) -> None:
    module = _load_cli(RUNNER_SCRIPT, "_capture_collision_io")
    paths = [
        (tmp_path / "metadata.json").resolve(),
        (tmp_path / "payload.json").resolve(),
        (tmp_path / "gate.json").resolve(),
    ]
    paths[1].write_bytes(b"foreign\n")

    with pytest.raises(FileExistsError):
        module._FinalizationCaptureFiles(
            metadata_path=paths[0],
            payload_path=paths[1],
            gate_path=paths[2],
        )

    assert not paths[0].exists()
    assert paths[1].read_bytes() == b"foreign\n"
    assert not paths[2].exists()
