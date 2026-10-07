from __future__ import annotations

import gc
import hashlib
import json
import os
import stat
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from mlx_vq.io import authenticated_artifacts as artifacts
from mlx_vq.io.authenticated_artifacts import (
    AuthenticatedFile,
    DescriptorSnapshot,
    PublicationCorruptError,
    PublicationUncertainError,
    authenticate_json,
    clone_or_copy_authenticated,
    publish_bytes_transactionally,
)


def test_authenticate_json_binds_exact_bytes_and_rejects_duplicate_and_nonfinite(
    tmp_path: Path,
) -> None:
    path = tmp_path / "authority.json"
    payload = b'{"z":1,"a":[true,null]}\n'
    path.write_bytes(payload)

    authenticated, value = authenticate_json(path, label="authority")
    try:
        assert value == {"z": 1, "a": [True, None]}
        assert authenticated.bytes == payload
        assert authenticated.sha256 == hashlib.sha256(payload).hexdigest()
        assert authenticated.size == len(payload)
        assert authenticated.path == path
    finally:
        authenticated.close()

    for invalid in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}'):
        path.write_bytes(invalid)
        with pytest.raises(ValueError, match="duplicate|finite|JSON"):
            authenticate_json(path, label="authority")


def test_authenticated_file_rejects_symlink_and_retarget_and_use_after_close(
    tmp_path: Path,
) -> None:
    path = tmp_path / "authority"
    path.write_bytes(b"trusted")
    link = tmp_path / "link"
    link.symlink_to(path.name)
    with pytest.raises(ValueError, match="symlink|following links"):
        AuthenticatedFile.open(link, label="authority")

    authenticated = AuthenticatedFile.open(path, label="authority")
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"trusted")
    os.replace(replacement, path)
    with pytest.raises(ValueError, match="retarget|identity"):
        authenticated.verify_visible()
    descriptor = authenticated.descriptor
    authenticated.close()
    authenticated.close()
    with pytest.raises(ValueError, match="closed"):
        _ = authenticated.bytes
    with pytest.raises(OSError):
        os.fstat(descriptor)


def test_symlink_opt_in_rejects_link_parent_rename_and_equivalent_recreation(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"trusted")
    link_parent = tmp_path / "links"
    link_parent.mkdir()
    link = link_parent / "authority"
    link.symlink_to(Path("../target"))
    authenticated = AuthenticatedFile.open(
        link,
        label="authority",
        allow_resolved_symlink=True,
    )
    moved = tmp_path / "moved-links"
    link_parent.rename(moved)
    link_parent.mkdir()
    link.symlink_to(Path("../target"))
    try:
        with pytest.raises(ValueError, match="symlink|ancestor|identity|retarget"):
            authenticated.verify_visible()
    finally:
        authenticated.close()


def test_symlink_opt_in_rejects_leaf_recreation_to_identical_target(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"trusted")
    link = tmp_path / "authority"
    link.symlink_to(target.name)
    authenticated = AuthenticatedFile.open(
        link,
        label="authority",
        allow_resolved_symlink=True,
    )
    link.unlink()
    link.symlink_to(target.name)
    try:
        with pytest.raises(ValueError, match="symlink|identity|retarget"):
            authenticated.verify_visible()
    finally:
        authenticated.close()


def test_authenticated_file_rejects_in_place_mutation(tmp_path: Path) -> None:
    path = tmp_path / "authority"
    path.write_bytes(b"trusted")
    authenticated = AuthenticatedFile.open(path, label="authority")
    path.write_bytes(b"changed")
    try:
        with pytest.raises(ValueError, match="identity changed"):
            authenticated.verify_unchanged()
    finally:
        authenticated.close()


@pytest.mark.parametrize("failure", ("missing", "permission", "raced-leaf"))
def test_authenticated_file_leaf_open_failures_never_leak_ancestor_descriptors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    path = tmp_path / "authority"
    if failure != "missing":
        path.write_bytes(b"trusted")
    real_open = artifacts._open_regular_no_follow

    if failure == "permission":
        def fail_permission(*_args, **_kwargs):
            raise ValueError("could not open authority without following links: denied")

        monkeypatch.setattr(artifacts, "_open_regular_no_follow", fail_permission)
    elif failure == "raced-leaf":
        replacement = tmp_path / "replacement"
        replacement.write_bytes(b"replacement")

        def race_leaf(candidate, *, label, dir_fd=None):
            path.unlink(missing_ok=True)
            path.symlink_to(replacement.name)
            try:
                return real_open(candidate, label=label, dir_fd=dir_fd)
            finally:
                path.unlink(missing_ok=True)
                path.write_bytes(b"trusted")

        monkeypatch.setattr(artifacts, "_open_regular_no_follow", race_leaf)

    before = len(os.listdir("/dev/fd"))
    for _ in range(64):
        with pytest.raises(ValueError, match="open|following links"):
            AuthenticatedFile.open(path, label="authority")
    gc.collect()
    assert len(os.listdir("/dev/fd")) == before


def test_descriptor_snapshot_exposes_live_dev_fd_paths_and_closes_idempotently(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    sources = {
        "first": AuthenticatedFile.open(first, label="first"),
        "second": AuthenticatedFile.open(second, label="second"),
    }
    snapshot = DescriptorSnapshot.create(
        sources,
        scratch_dir=tmp_path,
        allow_copy_fallback=True,
    )
    descriptors = snapshot.descriptors
    try:
        assert Path(snapshot.file_paths["first"]).read_bytes() == b"one"
        assert Path(snapshot.file_paths["second"]).read_bytes() == b"two"
    finally:
        snapshot.close()
        snapshot.close()
        for source in sources.values():
            source.close()
    assert snapshot.file_paths == {}
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_descriptor_snapshot_finalizer_releases_descriptors(tmp_path: Path) -> None:
    path = tmp_path / "source"
    path.write_bytes(b"payload")
    source = AuthenticatedFile.open(path, label="source")
    snapshot = DescriptorSnapshot.create(
        {"source": source},
        scratch_dir=tmp_path,
        allow_copy_fallback=True,
    )
    descriptors = snapshot.descriptors
    del snapshot
    gc.collect()
    source.close()
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_descriptor_snapshot_revokes_retained_mapping_alias_on_close(
    tmp_path: Path,
) -> None:
    path = tmp_path / "source"
    path.write_bytes(b"payload")
    source = AuthenticatedFile.open(path, label="source")
    snapshot = DescriptorSnapshot.create(
        {"source": source},
        scratch_dir=tmp_path,
        allow_copy_fallback=True,
    )
    alias = snapshot.file_paths
    snapshot.close()
    source.close()
    assert alias == {}


def test_descriptor_snapshot_combine_revokes_component_and_combined_aliases(
    tmp_path: Path,
) -> None:
    sources: list[AuthenticatedFile] = []
    components: list[DescriptorSnapshot] = []
    aliases = []
    for name in ("first", "second"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        source = AuthenticatedFile.open(path, label=name)
        sources.append(source)
        component = DescriptorSnapshot.create(
            {name: source},
            scratch_dir=tmp_path,
            allow_copy_fallback=True,
        )
        components.append(component)
        aliases.append(component.file_paths)
    combined = DescriptorSnapshot.combine(tuple(components))
    combined_alias = combined.file_paths
    try:
        assert aliases == [{}, {}]
        assert set(combined_alias) == {"first", "second"}
    finally:
        combined.close()
        for source in sources:
            source.close()
    assert combined_alias == {}


def test_descriptor_snapshot_partial_failure_closes_retained_fds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    sources = {
        "first": AuthenticatedFile.open(first, label="first"),
        "second": AuthenticatedFile.open(second, label="second"),
    }
    real = clone_or_copy_authenticated
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected copy failure")
        return real(*args, **kwargs)

    monkeypatch.setattr(
        "mlx_vq.io.authenticated_artifacts.clone_or_copy_authenticated", fail_second
    )
    before = len(os.listdir("/dev/fd"))
    with pytest.raises(RuntimeError, match="injected copy failure"):
        DescriptorSnapshot.create(
            sources,
            scratch_dir=tmp_path,
            allow_copy_fallback=True,
        )
    gc.collect()
    assert len(os.listdir("/dev/fd")) <= before
    for source in sources.values():
        source.close()


def test_clone_fallback_is_explicit_and_space_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "source"
    source_path.write_bytes(b"payload")
    source = AuthenticatedFile.open(source_path, label="source")
    monkeypatch.setattr(
        "mlx_vq.io.authenticated_artifacts._try_fclonefileat",
        lambda *_args, **_kwargs: False,
    )
    try:
        with pytest.raises(ValueError, match="not explicitly enabled"):
            clone_or_copy_authenticated(
                source,
                tmp_path / "disabled",
                allow_copy_fallback=False,
                required_free_bytes=7,
            )
        monkeypatch.setattr(
            "mlx_vq.io.authenticated_artifacts.os.fstatvfs",
            lambda _descriptor: type(
                "Filesystem", (), {"f_bavail": 6, "f_frsize": 1}
            )(),
        )
        with pytest.raises(ValueError, match="requires 7 free bytes"):
            clone_or_copy_authenticated(
                source,
                tmp_path / "bounded",
                allow_copy_fallback=True,
                required_free_bytes=7,
            )
    finally:
        source.close()


def test_clone_cow_success_uses_independent_authenticated_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "source"
    source_path.write_bytes(b"payload")
    source = AuthenticatedFile.open(source_path, label="source")

    def simulate_clone(
        descriptor: int,
        destination: Path,
        *,
        parent_descriptor: int,
    ) -> bool:
        output = os.open(
            destination.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=parent_descriptor,
        )
        try:
            os.write(output, os.pread(descriptor, source.size, 0))
        finally:
            os.close(output)
        return True

    monkeypatch.setattr(artifacts, "_try_fclonefileat", simulate_clone)
    destination = tmp_path / "snapshot"
    try:
        clone_or_copy_authenticated(
            source,
            destination,
            allow_copy_fallback=False,
            required_free_bytes=source.size,
        )
        assert destination.read_bytes() == b"payload"
        assert destination.stat().st_ino != source_path.stat().st_ino
        assert destination.stat().st_mode & 0o777 == 0o400
    finally:
        source.close()


def test_transactional_publication_is_exact_private_and_no_overwrite(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "manifest.json"
    payload = json.dumps({"a": 1}, sort_keys=True).encode() + b"\n"
    result = publish_bytes_transactionally(destination, payload)
    assert destination.read_bytes() == payload
    assert result.sha256 == hashlib.sha256(payload).hexdigest()
    assert destination.stat().st_mode & 0o777 == 0o400
    with pytest.raises(FileExistsError):
        publish_bytes_transactionally(destination, b"replacement")


def test_transactional_publication_rejects_symlink_ancestor(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink|ancestor"):
        publish_bytes_transactionally(alias / "manifest.json", b"{}\n")


def test_authentication_rejects_mutation_during_descriptor_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "authority"
    path.write_bytes(b"trusted")
    real_read = artifacts.os.read
    mutated = False

    def mutate_after_read(descriptor: int, size: int) -> bytes:
        nonlocal mutated
        chunk = real_read(descriptor, size)
        if chunk and not mutated:
            path.write_bytes(b"changed")
            mutated = True
        return chunk

    monkeypatch.setattr(artifacts.os, "read", mutate_after_read)
    with pytest.raises(ValueError, match="identity changed"):
        AuthenticatedFile.open(path, label="authority")


def test_clone_copy_interruption_removes_only_owned_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "source"
    source_path.write_bytes(b"payload")
    source = AuthenticatedFile.open(source_path, label="source")
    monkeypatch.setattr(
        artifacts, "_try_fclonefileat", lambda *_args, **_kwargs: False
    )
    monkeypatch.setattr(artifacts.os, "pread", lambda *_args: b"")
    destination = tmp_path / "snapshot"
    try:
        with pytest.raises(ValueError, match="ended while"):
            clone_or_copy_authenticated(
                source,
                destination,
                allow_copy_fallback=True,
                required_free_bytes=source.size,
            )
        assert not destination.exists()
    finally:
        source.close()


def test_clone_copy_rejects_source_retarget_and_cleans_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "source"
    source_path.write_bytes(b"payload")
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"payload")
    source = AuthenticatedFile.open(source_path, label="source")
    monkeypatch.setattr(
        artifacts, "_try_fclonefileat", lambda *_args, **_kwargs: False
    )
    real_pread = artifacts.os.pread
    swapped = False

    def swap_then_read(*args):
        nonlocal swapped
        chunk = real_pread(*args)
        if not swapped:
            os.replace(replacement, source_path)
            swapped = True
        return chunk

    monkeypatch.setattr(artifacts.os, "pread", swap_then_read)
    destination = tmp_path / "snapshot"
    try:
        with pytest.raises(ValueError, match="identity changed|retargeted"):
            clone_or_copy_authenticated(
                source,
                destination,
                allow_copy_fallback=True,
                required_free_bytes=source.size,
            )
        assert not destination.exists()
    finally:
        source.close()


def test_descriptor_snapshot_passes_only_remaining_copy_bytes(tmp_path: Path) -> None:
    paths = [tmp_path / "one", tmp_path / "two"]
    paths[0].write_bytes(b"111")
    paths[1].write_bytes(b"22")
    sources = {
        path.name: AuthenticatedFile.open(path, label=path.name) for path in paths
    }
    required: list[int] = []

    def record(source, destination, **kwargs):
        required.append(kwargs["required_free_bytes"])
        return clone_or_copy_authenticated(source, destination, **kwargs)

    snapshot = DescriptorSnapshot.create(
        sources,
        scratch_dir=tmp_path,
        allow_copy_fallback=True,
        clone_function=record,
    )
    try:
        assert required == [5, 2]
    finally:
        snapshot.close()
        for source in sources.values():
            source.close()


def test_snapshot_partial_failure_preserves_primary_over_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "source"
    source_path.write_bytes(b"payload")
    source = AuthenticatedFile.open(source_path, label="source")

    def fail_copy(*_args, **_kwargs):
        raise RuntimeError("primary snapshot failure")

    monkeypatch.setattr(
        artifacts.tempfile.TemporaryDirectory,
        "cleanup",
        lambda _self: (_ for _ in ()).throw(OSError("cleanup failure")),
    )
    try:
        with pytest.raises(RuntimeError, match="primary snapshot failure"):
            DescriptorSnapshot.create(
                {"source": source},
                scratch_dir=tmp_path,
                allow_copy_fallback=True,
                clone_function=fail_copy,
            )
    finally:
        source.close()


def test_publication_rejects_full_ancestor_aba_before_syscall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    moved = tmp_path / "moved"

    def replace_parent(*_args):
        parent.rename(moved)
        parent.mkdir()

    monkeypatch.setattr(artifacts, "_before_publication_syscall", replace_parent)
    with pytest.raises(ValueError, match="ancestor identity changed"):
        publish_bytes_transactionally(parent / "manifest", b"trusted")
    assert not (parent / "manifest").exists()


def test_publication_rejects_staging_substitution_before_syscall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "manifest"

    def substitute(_target: Path, parent: int, staging: str) -> None:
        os.rename(staging, f"{staging}.authentic", src_dir_fd=parent, dst_dir_fd=parent)
        descriptor = os.open(
            staging,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=parent,
        )
        os.write(descriptor, b"attacker")
        os.close(descriptor)

    monkeypatch.setattr(artifacts, "_before_publication_syscall", substitute)
    with pytest.raises(ValueError, match="source was substituted"):
        publish_bytes_transactionally(destination, b"trusted")
    assert not destination.exists()


def test_publication_rejects_same_inode_staging_mutation_before_syscall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "manifest"

    def mutate(_target: Path, parent: int, staging: str) -> None:
        descriptor = os.open(staging, os.O_WRONLY | os.O_TRUNC, dir_fd=parent)
        os.write(descriptor, b"attacker")
        os.close(descriptor)

    monkeypatch.setattr(artifacts, "_before_publication_syscall", mutate)
    with pytest.raises(ValueError, match="source changed before commit"):
        publish_bytes_transactionally(destination, b"trusted", mode=0o600)
    assert not destination.exists()


def test_publication_reports_corrupt_post_syscall_destination_without_deleting_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "manifest"
    attacker = tmp_path / "attacker"
    attacker.write_bytes(b"attacker")

    def substitute(target: Path, parent: int, _staging: str) -> None:
        os.rename(attacker.name, target.name, src_dir_fd=parent, dst_dir_fd=parent)

    monkeypatch.setattr(artifacts, "_after_publication_syscall", substitute)
    with pytest.raises(PublicationCorruptError, match="do not match"):
        publish_bytes_transactionally(destination, b"trusted")
    assert destination.read_bytes() == b"attacker"
    forensic = list(tmp_path.glob(".manifest.partial-*"))
    assert len(forensic) == 1
    assert forensic[0].read_bytes() == b"trusted"


def test_post_publication_hook_failure_reconciles_authentic_final_as_uncertain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "manifest"

    def fail_after(*_args):
        raise RuntimeError("injected post-publication failure")

    monkeypatch.setattr(artifacts, "_after_publication_syscall", fail_after)
    with pytest.raises(
        PublicationUncertainError,
        match="durability.*uncertain|committed.*uncertain",
    ):
        publish_bytes_transactionally(destination, b"trusted")
    assert destination.read_bytes() == b"trusted"


def test_post_publication_ancestor_replacement_is_corrupt_not_generic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    moved = tmp_path / "moved"

    def replace_after(*_args):
        parent.rename(moved)
        parent.mkdir()

    monkeypatch.setattr(artifacts, "_after_publication_syscall", replace_after)
    with pytest.raises(PublicationCorruptError, match="could not be authenticated"):
        publish_bytes_transactionally(parent / "manifest", b"trusted")
    assert not (parent / "manifest").exists()
    assert (moved / "manifest").read_bytes() == b"trusted"


def test_allow_replace_post_publication_failure_is_uncertain_and_preserves_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "manifest"
    destination.write_bytes(b"old")
    authenticated = AuthenticatedFile.open(destination, label="destination")
    expected = artifacts.ReplacementAuthority.from_authenticated(authenticated)
    authenticated.close()

    def fail_after(*_args):
        raise RuntimeError("injected replace post-publication failure")

    monkeypatch.setattr(artifacts, "_after_publication_syscall", fail_after)
    with pytest.raises(PublicationUncertainError, match="committed.*uncertain"):
        publish_bytes_transactionally(
            destination,
            b"trusted",
            allow_replace=True,
            expected_destination=expected,
            mode=0o600,
        )
    assert destination.read_bytes() == b"trusted"


def test_replace_requires_exact_authenticated_destination_identity(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "manifest"
    destination.write_bytes(b"old")
    with pytest.raises(ValueError, match="expected.*destination|replacement authority"):
        publish_bytes_transactionally(
            destination,
            b"trusted",
            allow_replace=True,
            mode=0o600,
        )
    assert destination.read_bytes() == b"old"


def test_replace_rejects_destination_swap_after_expected_identity_capture(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "manifest"
    destination.write_bytes(b"old")
    authenticated = AuthenticatedFile.open(destination, label="destination")
    expected = artifacts.ReplacementAuthority.from_authenticated(authenticated)
    authenticated.close()
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"attacker")
    os.replace(replacement, destination)
    with pytest.raises(ValueError, match="destination.*identity|replacement authority"):
        publish_bytes_transactionally(
            destination,
            b"trusted",
            expected_destination=expected,
            mode=0o600,
        )
    assert destination.read_bytes() == b"attacker"


def test_replacement_exchange_preserves_swap_injected_after_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "manifest"
    destination.write_bytes(b"old")
    authenticated = AuthenticatedFile.open(destination, label="destination")
    expected = artifacts.ReplacementAuthority.from_authenticated(authenticated)
    authenticated.close()
    attacker = tmp_path / "attacker"
    attacker.write_bytes(b"attacker")
    saved_expected = tmp_path / "saved-expected"

    def inject_swap(*_args):
        os.rename(destination, saved_expected)
        os.rename(attacker, destination)

    monkeypatch.setattr(
        artifacts,
        "_before_replacement_exchange",
        inject_swap,
        raising=False,
    )
    with pytest.raises(PublicationCorruptError):
        publish_bytes_transactionally(
            destination,
            b"trusted",
            expected_destination=expected,
            mode=0o600,
        )
    assert destination.read_bytes() == b"trusted"
    assert saved_expected.read_bytes() == b"old"
    forensic = list(tmp_path.glob(".manifest.partial-*"))
    assert len(forensic) == 1
    assert forensic[0].read_bytes() == b"attacker"


def test_replacement_exchange_rejects_post_exchange_displaced_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "manifest"
    destination.write_bytes(b"old")
    authenticated = AuthenticatedFile.open(destination, label="destination")
    expected = artifacts.ReplacementAuthority.from_authenticated(authenticated)
    authenticated.close()
    attacker = tmp_path / "attacker"
    attacker.write_bytes(b"attacker")
    saved_expected = tmp_path / "saved-expected"

    def substitute_displaced(_target: Path, parent: int, staging: str) -> None:
        os.rename(
            staging,
            saved_expected.name,
            src_dir_fd=parent,
            dst_dir_fd=parent,
        )
        os.rename(
            attacker.name,
            staging,
            src_dir_fd=parent,
            dst_dir_fd=parent,
        )

    monkeypatch.setattr(
        artifacts, "_after_publication_syscall", substitute_displaced
    )
    with pytest.raises(PublicationCorruptError, match="displaced replacement"):
        publish_bytes_transactionally(
            destination,
            b"trusted",
            expected_destination=expected,
            mode=0o600,
        )
    assert destination.read_bytes() == b"trusted"
    assert saved_expected.read_bytes() == b"old"
    forensic = list(tmp_path.glob(".manifest.partial-*"))
    assert len(forensic) == 1
    assert forensic[0].read_bytes() == b"attacker"


def test_replacement_exchange_directory_fsync_failure_preserves_displaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "manifest"
    destination.write_bytes(b"old")
    authenticated = AuthenticatedFile.open(destination, label="destination")
    expected = artifacts.ReplacementAuthority.from_authenticated(authenticated)
    authenticated.close()
    real_fsync = artifacts.os.fsync

    def fail_directory_fsync(descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("injected replacement directory fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(artifacts.os, "fsync", fail_directory_fsync)
    with pytest.raises(
        PublicationUncertainError,
        match="durability.*uncertain|committed.*uncertain",
    ):
        publish_bytes_transactionally(
            destination,
            b"trusted",
            expected_destination=expected,
            mode=0o600,
        )
    assert destination.read_bytes() == b"trusted"
    forensic = list(tmp_path.glob(".manifest.partial-*"))
    assert len(forensic) == 1
    assert forensic[0].read_bytes() == b"old"


def test_successful_replacement_never_unlinks_displaced_and_returns_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "manifest"
    destination.write_bytes(b"old")
    authenticated = AuthenticatedFile.open(destination, label="destination")
    expected = artifacts.ReplacementAuthority.from_authenticated(authenticated)
    authenticated.close()
    real_unlink = artifacts.os.unlink
    displaced_unlinks: list[str] = []

    def forbid_displaced_unlink(path, *args, **kwargs):
        if str(path).startswith(".manifest.partial-"):
            displaced_unlinks.append(str(path))
            raise AssertionError("publication must retain displaced evidence")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(artifacts.os, "unlink", forbid_displaced_unlink)
    published = publish_bytes_transactionally(
        destination,
        b"trusted",
        expected_destination=expected,
        mode=0o600,
    )
    assert displaced_unlinks == []
    assert destination.read_bytes() == b"trusted"
    assert published.displaced_path is not None
    assert published.displaced_path.read_bytes() == b"old"
    assert published.displaced_identity is not None
    assert published.displaced_identity.inode == expected.identity.inode
    assert published.displaced_sha256 == expected.sha256


def test_displaced_cleanup_is_explicit_nondestructive_archival(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "manifest"
    destination.write_bytes(b"old")
    authenticated = AuthenticatedFile.open(destination, label="destination")
    expected = artifacts.ReplacementAuthority.from_authenticated(authenticated)
    authenticated.close()
    published = publish_bytes_transactionally(
        destination,
        b"trusted",
        expected_destination=expected,
        mode=0o600,
    )
    displaced = published.displaced_path
    assert displaced is not None and displaced.read_bytes() == b"old"
    archive = tmp_path / "forensics"
    archive.mkdir(mode=0o700)
    archived = artifacts.archive_displaced_artifact(published, archive)
    assert archived is not None
    assert archived.read_bytes() == b"old"
    assert not displaced.exists()
    assert destination.read_bytes() == b"trusted"


def test_replacement_exchange_reauthenticates_displaced_immediately_before_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "manifest"
    destination.write_bytes(b"old")
    authenticated = AuthenticatedFile.open(destination, label="destination")
    expected = artifacts.ReplacementAuthority.from_authenticated(authenticated)
    authenticated.close()
    attacker = tmp_path / "attacker"
    attacker.write_bytes(b"attacker")
    saved_expected = tmp_path / "saved-expected"

    def substitute_before_evidence_return(
        _target: Path,
        parent: int,
        staging: str,
    ) -> None:
        os.rename(
            staging,
            saved_expected.name,
            src_dir_fd=parent,
            dst_dir_fd=parent,
        )
        os.rename(
            attacker.name,
            staging,
            src_dir_fd=parent,
            dst_dir_fd=parent,
        )

    monkeypatch.setattr(
        artifacts,
        "_before_displaced_evidence_return",
        substitute_before_evidence_return,
    )
    with pytest.raises(PublicationCorruptError, match="before cleanup"):
        publish_bytes_transactionally(
            destination,
            b"trusted",
            expected_destination=expected,
            mode=0o600,
        )
    assert destination.read_bytes() == b"trusted"
    assert saved_expected.read_bytes() == b"old"
    forensic = list(tmp_path.glob(".manifest.partial-*"))
    assert len(forensic) == 1
    assert forensic[0].read_bytes() == b"attacker"


def test_file_publication_reauthenticates_original_source_immediately_before_syscall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"trusted")
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"changed")
    destination = tmp_path / "manifest"

    def swap_source(*_args):
        os.replace(replacement, source)

    monkeypatch.setattr(artifacts, "_before_publication_syscall", swap_source)
    with pytest.raises(ValueError, match="source.*identity|retarget|changed"):
        artifacts.publish_file_transactionally(source, destination)
    assert not destination.exists()


def test_allow_replace_corrupt_destination_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "manifest"
    destination.write_bytes(b"old")
    authenticated = AuthenticatedFile.open(destination, label="destination")
    expected = artifacts.ReplacementAuthority.from_authenticated(authenticated)
    authenticated.close()
    attacker = tmp_path / "attacker"
    attacker.write_bytes(b"attacker")

    def substitute(target: Path, parent: int, _staging: str) -> None:
        os.rename(attacker.name, target.name, src_dir_fd=parent, dst_dir_fd=parent)

    monkeypatch.setattr(artifacts, "_after_publication_syscall", substitute)
    with pytest.raises(PublicationCorruptError, match="corrupt|do not match"):
        publish_bytes_transactionally(
            destination,
            b"trusted",
            allow_replace=True,
            expected_destination=expected,
            mode=0o600,
        )
    assert destination.read_bytes() == b"attacker"
    forensic = list(tmp_path.glob(".manifest.partial-*"))
    assert len(forensic) == 1
    assert forensic[0].read_bytes() == b"old"


def test_post_publication_staging_cleanup_failure_is_uncertain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "manifest"
    real_unlink = artifacts.os.unlink

    def fail_staging_cleanup(path, *args, **kwargs):
        if str(path).startswith(".manifest.partial-"):
            raise OSError("injected post-publication cleanup failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(artifacts.os, "unlink", fail_staging_cleanup)
    with pytest.raises(PublicationUncertainError, match="committed.*uncertain"):
        publish_bytes_transactionally(destination, b"trusted")
    assert destination.read_bytes() == b"trusted"
    assert len(list(tmp_path.glob(".manifest.partial-*"))) == 1


def test_file_publication_source_drift_after_commit_is_corrupt_not_generic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"trusted")
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"changed")
    destination = tmp_path / "manifest"
    real_publish = artifacts.publish_bytes_transactionally

    def publish_then_retarget(*args, **kwargs):
        result = real_publish(*args, **kwargs)
        os.replace(replacement, source)
        return result

    monkeypatch.setattr(
        artifacts, "publish_bytes_transactionally", publish_then_retarget
    )
    with pytest.raises(PublicationCorruptError, match="source.*changed"):
        artifacts.publish_file_transactionally(source, destination)
    assert destination.read_bytes() == b"trusted"


def test_destination_open_failure_after_publication_is_corrupt_and_preserves_forensics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "manifest"
    real_open = artifacts.os.open
    published = False

    def mark_published(*_args):
        nonlocal published
        published = True

    def fail_destination_open(path, flags, *args, **kwargs):
        if published and path == destination.name and flags & os.O_ACCMODE == os.O_RDONLY:
            raise PermissionError("injected destination open failure")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(artifacts, "_after_publication_syscall", mark_published)
    monkeypatch.setattr(artifacts.os, "open", fail_destination_open)
    with pytest.raises(PublicationCorruptError, match="could not be authenticated"):
        publish_bytes_transactionally(destination, b"trusted")
    assert destination.read_bytes() == b"trusted"
    assert len(list(tmp_path.glob(".manifest.partial-*"))) == 1


def test_publication_fsync_failure_is_uncertain_and_keeps_authentic_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "manifest"
    real_fsync = artifacts.os.fsync

    def fail_directory_fsync(descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("injected directory fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(artifacts.os, "fsync", fail_directory_fsync)
    with pytest.raises(PublicationUncertainError, match="durability is uncertain"):
        publish_bytes_transactionally(destination, b"trusted")
    assert destination.read_bytes() == b"trusted"


def test_post_durability_destination_replacement_is_corrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "manifest"
    attacker = tmp_path / "attacker"
    attacker.write_bytes(b"attacker")
    real_fsync = artifacts.os.fsync
    swapped = False

    def replace_after_directory_fsync(descriptor: int) -> None:
        nonlocal swapped
        real_fsync(descriptor)
        if stat.S_ISDIR(os.fstat(descriptor).st_mode) and not swapped:
            os.replace(attacker, destination)
            swapped = True

    monkeypatch.setattr(artifacts.os, "fsync", replace_after_directory_fsync)
    with pytest.raises(PublicationCorruptError, match="corrupt|do not match"):
        publish_bytes_transactionally(destination, b"trusted")
    assert swapped is True
    assert destination.read_bytes() == b"attacker"


def test_publication_cleanup_failure_does_not_mask_primary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "manifest"

    def fail_before(*_args):
        raise RuntimeError("primary publication failure")

    monkeypatch.setattr(artifacts, "_before_publication_syscall", fail_before)
    real_unlink = artifacts.os.unlink

    def fail_cleanup(path, *args, **kwargs):
        if str(path).startswith(".manifest.partial-"):
            raise OSError("cleanup failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(artifacts.os, "unlink", fail_cleanup)
    with pytest.raises(RuntimeError, match="primary publication failure"):
        publish_bytes_transactionally(destination, b"trusted")


def test_publication_no_overwrite_is_exact_under_concurrency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "manifest"
    barrier = threading.Barrier(2)

    def synchronize(*_args):
        barrier.wait(timeout=5)

    monkeypatch.setattr(artifacts, "_before_publication_syscall", synchronize)

    def publish(payload: bytes):
        try:
            return publish_bytes_transactionally(destination, payload)
        except BaseException as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(publish, (b"first", b"second")))
    successes = [result for result in results if not isinstance(result, BaseException)]
    failures = [result for result in results if isinstance(result, BaseException)]
    assert len(successes) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], FileExistsError)
    assert destination.read_bytes() in {b"first", b"second"}


def test_unsafe_dotdot_publication_path_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsafe component"):
        publish_bytes_transactionally(tmp_path / "child" / ".." / "manifest", b"x")
