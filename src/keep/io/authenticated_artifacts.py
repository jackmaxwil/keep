"""Small descriptor-bound primitives for authenticated local artifacts.

This module is deliberately headless: it imports neither MLX nor any model
code.  Paths are used only to acquire and revalidate descriptors; trusted
bytes always come from an already-open regular-file descriptor.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import stat
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any


_READ_CHUNK = 16 * 1024 * 1024


class PublicationUncertainError(RuntimeError):
    """The destination is committed but its directory fsync was not proven."""


class PublicationCorruptError(RuntimeError):
    """A pathname publication occurred but the destination is not authentic."""


@dataclass(frozen=True, slots=True)
class FileIdentity:
    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int
    ctime_ns: int
    link_count: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> "FileIdentity":
        return cls(
            device=value.st_dev,
            inode=value.st_ino,
            mode=value.st_mode,
            size=value.st_size,
            mtime_ns=value.st_mtime_ns,
            ctime_ns=value.st_ctime_ns,
            link_count=value.st_nlink,
        )


@dataclass(frozen=True, slots=True)
class ReplacementAuthority:
    """Exact authenticated authority required to replace one destination."""

    identity: FileIdentity
    sha256: str

    @classmethod
    def from_authenticated(cls, value: "AuthenticatedFile") -> "ReplacementAuthority":
        value.verify_visible()
        return cls(identity=value.identity, sha256=value.sha256)


def _directory_identity(descriptor: int) -> tuple[int, int, int]:
    value = os.fstat(descriptor)
    return value.st_dev, value.st_ino, value.st_mode


@dataclass(slots=True)
class _AncestorAuthority:
    """Retained root-to-parent descriptors plus their visible name linkage."""

    absolute_parent: Path
    descriptors: tuple[int, ...]
    names: tuple[str, ...]
    identities: tuple[tuple[int, int, int], ...]
    _closed: bool = False

    @classmethod
    def open(cls, leaf: Path) -> "_AncestorAuthority":
        if ".." in leaf.parts or leaf.name in {"", ".", ".."}:
            raise ValueError(f"artifact path contains an unsafe component: {leaf}")
        absolute_parent = leaf.absolute().parent
        components = absolute_parent.parts[1:]
        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptors: list[int] = []
        names: list[str] = []
        identities: list[tuple[int, int, int]] = []
        try:
            current = os.open(os.sep, flags)
            descriptors.append(current)
            identities.append(_directory_identity(current))
            for component in components:
                try:
                    child = os.open(component, flags, dir_fd=current)
                except OSError as error:
                    raise ValueError(
                        f"artifact ancestor must be a no-follow directory: "
                        f"{absolute_parent}"
                    ) from error
                identity = _directory_identity(child)
                if not stat.S_ISDIR(identity[2]):
                    os.close(child)
                    raise ValueError(
                        f"artifact ancestor must be a directory: {absolute_parent}"
                    )
                descriptors.append(child)
                names.append(component)
                identities.append(identity)
                current = child
            authority = cls(
                absolute_parent=absolute_parent,
                descriptors=tuple(descriptors),
                names=tuple(names),
                identities=tuple(identities),
            )
            authority.verify()
            return authority
        except BaseException:
            for descriptor in reversed(descriptors):
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            raise

    @property
    def parent_descriptor(self) -> int:
        if self._closed:
            raise ValueError("artifact ancestor authority is closed")
        return self.descriptors[-1]

    def verify(self) -> None:
        if self._closed:
            raise ValueError("artifact ancestor authority is closed")
        for index, descriptor in enumerate(self.descriptors):
            if _directory_identity(descriptor) != self.identities[index]:
                raise ValueError(
                    f"artifact ancestor identity changed: {self.absolute_parent}"
                )
            if index:
                try:
                    visible = os.stat(
                        self.names[index - 1],
                        dir_fd=self.descriptors[index - 1],
                        follow_symlinks=False,
                    )
                except OSError as error:
                    raise ValueError(
                        f"artifact ancestor identity changed: {self.absolute_parent}"
                    ) from error
                if (
                    visible.st_dev,
                    visible.st_ino,
                    visible.st_mode,
                ) != self.identities[index]:
                    raise ValueError(
                        f"artifact ancestor identity changed: {self.absolute_parent}"
                    )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for descriptor in reversed(self.descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass
        self.descriptors = ()


@dataclass(slots=True)
class _SymlinkAuthority:
    """Retained lexical link authority, distinct from its resolved target."""

    path: Path
    ancestors: _AncestorAuthority
    identity: FileIdentity
    link_text: str
    resolved_target: Path
    _closed: bool = False

    @classmethod
    def open(cls, path: Path, *, label: str) -> "_SymlinkAuthority":
        ancestors = _AncestorAuthority.open(path)
        try:
            visible = os.stat(
                path.name,
                dir_fd=ancestors.parent_descriptor,
                follow_symlinks=False,
            )
            if not stat.S_ISLNK(visible.st_mode):
                raise ValueError(f"{label} must be the authenticated symlink leaf")
            link_text = os.readlink(path.name, dir_fd=ancestors.parent_descriptor)
            raw_target = Path(link_text)
            target = (
                raw_target
                if raw_target.is_absolute()
                else ancestors.absolute_parent / raw_target
            ).resolve(strict=True)
            authority = cls(
                path=path,
                ancestors=ancestors,
                identity=FileIdentity.from_stat(visible),
                link_text=link_text,
                resolved_target=target,
            )
            authority.verify()
            return authority
        except BaseException:
            ancestors.close()
            raise

    def verify(self) -> None:
        if self._closed:
            raise ValueError("symlink authority is closed")
        self.ancestors.verify()
        try:
            visible = os.stat(
                self.path.name,
                dir_fd=self.ancestors.parent_descriptor,
                follow_symlinks=False,
            )
            link_text = os.readlink(
                self.path.name,
                dir_fd=self.ancestors.parent_descriptor,
            )
        except OSError as error:
            raise ValueError(
                f"symlink identity changed during authenticated inspection: {self.path}"
            ) from error
        if FileIdentity.from_stat(visible) != self.identity or link_text != self.link_text:
            raise ValueError(
                f"symlink identity changed during authenticated inspection: {self.path}"
            )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.ancestors.close()


def _open_regular_no_follow(
    path: Path | str,
    *,
    label: str,
    dir_fd: int | None = None,
) -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, dir_fd=dir_fd)
    except OSError as error:
        raise ValueError(
            f"could not open {label} without following links: {error}"
        ) from error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink == 0:
            raise ValueError(f"{label} must be a regular file")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def open_regular_no_follow(path: str | Path, *, label: str) -> int:
    """Compatibility descriptor open with the shared no-follow policy."""

    candidate = Path(path)
    ancestors = _AncestorAuthority.open(candidate)
    try:
        descriptor = _open_regular_no_follow(
            candidate.name,
            label=label,
            dir_fd=ancestors.parent_descriptor,
        )
        ancestors.verify()
        return descriptor
    finally:
        ancestors.close()


def _read_descriptor(descriptor: int) -> tuple[bytes, str]:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    digest = hashlib.sha256()
    while True:
        chunk = os.read(descriptor, _READ_CHUNK)
        if not chunk:
            break
        chunks.append(chunk)
        digest.update(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return b"".join(chunks), digest.hexdigest()


@dataclass(frozen=True, slots=True)
class AuthenticatedFile:
    """An exact byte sequence retained behind its authenticated descriptor."""

    path: Path
    label: str
    identity: FileIdentity
    sha256: str
    _bytes: bytes = field(repr=False)
    _descriptor: int = field(repr=False)
    _opened_path: Path = field(repr=False)
    _ancestors: _AncestorAuthority = field(repr=False, compare=False)
    _link_authority: _SymlinkAuthority | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _resolved_symlink: bool = field(default=False, repr=False, compare=False)
    _closed: bool = field(default=False, repr=False, compare=False)

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        label: str,
        allow_resolved_symlink: bool = False,
    ) -> "AuthenticatedFile":
        candidate = Path(path)
        resolved_symlink = candidate.is_symlink()
        if resolved_symlink and not allow_resolved_symlink:
            raise ValueError(f"{label} must not be a symlink")
        link_authority: _SymlinkAuthority | None = None
        try:
            if resolved_symlink:
                link_authority = _SymlinkAuthority.open(candidate, label=label)
                opened_path = link_authority.resolved_target
            else:
                opened_path = candidate
        except OSError as error:
            raise ValueError(f"could not resolve {label}: {error}") from error
        try:
            ancestors = _AncestorAuthority.open(opened_path)
        except BaseException:
            if link_authority is not None:
                link_authority.close()
            raise
        try:
            descriptor = _open_regular_no_follow(
                opened_path.name,
                label=label,
                dir_fd=ancestors.parent_descriptor,
            )
        except BaseException:
            ancestors.close()
            if link_authority is not None:
                link_authority.close()
            raise
        try:
            before = FileIdentity.from_stat(os.fstat(descriptor))
            payload, digest = _read_descriptor(descriptor)
            after = FileIdentity.from_stat(os.fstat(descriptor))
            if after != before:
                raise ValueError(f"{label} identity changed while it was read")
            ancestors.verify()
            authenticated = cls(
                path=candidate,
                label=label,
                identity=before,
                sha256=digest,
                _bytes=payload,
                _descriptor=descriptor,
                _opened_path=opened_path.absolute(),
                _ancestors=ancestors,
                _link_authority=link_authority,
                _resolved_symlink=resolved_symlink,
            )
            authenticated.verify_visible()
            return authenticated
        except BaseException:
            os.close(descriptor)
            ancestors.close()
            if link_authority is not None:
                link_authority.close()
            raise

    def _require_open(self) -> None:
        if self._closed:
            raise ValueError(f"{self.label} authenticated file is closed")

    @property
    def bytes(self) -> bytes:
        self._require_open()
        return self._bytes

    @property
    def descriptor(self) -> int:
        self._require_open()
        return self._descriptor

    @property
    def device(self) -> int:
        return self.identity.device

    @property
    def inode(self) -> int:
        return self.identity.inode

    @property
    def mode(self) -> int:
        return self.identity.mode

    @property
    def size(self) -> int:
        return self.identity.size

    def verify_unchanged(self) -> None:
        self._require_open()
        current = FileIdentity.from_stat(os.fstat(self._descriptor))
        if current != self.identity:
            raise ValueError(f"{self.label} identity changed during authenticated use")

    def verify_visible(self) -> None:
        self.verify_unchanged()
        if self._link_authority is not None:
            self._link_authority.verify()
        self._ancestors.verify()
        if self._resolved_symlink:
            try:
                if self.path.resolve(strict=True).absolute() != self._opened_path:
                    raise ValueError(
                        f"{self.label} was retargeted during authenticated inspection"
                    )
            except OSError as error:
                raise ValueError(
                    f"{self.label} was retargeted during authenticated inspection"
                ) from error
        reopened = _open_regular_no_follow(
            self._opened_path.name,
            label=self.label,
            dir_fd=self._ancestors.parent_descriptor,
        )
        try:
            current = FileIdentity.from_stat(os.fstat(reopened))
            if current != self.identity:
                raise ValueError(
                    f"{self.label} was retargeted during authenticated inspection"
                )
        finally:
            os.close(reopened)

    def close(self) -> None:
        if self._closed:
            return
        descriptor = self._descriptor
        object.__setattr__(self, "_closed", True)
        object.__setattr__(self, "_descriptor", -1)
        try:
            os.close(descriptor)
        finally:
            try:
                self._ancestors.close()
            finally:
                if self._link_authority is not None:
                    self._link_authority.close()

    def __enter__(self) -> "AuthenticatedFile":
        self._require_open()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except (AttributeError, OSError):
            pass


def _reject_constant(value: str) -> Any:
    raise ValueError(f"JSON number must be finite: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json_bytes(
    payload: bytes,
    *,
    label: str,
    loads: Callable[..., Any] = json.loads,
) -> dict[str, Any]:
    try:
        value = loads(
            payload,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"could not parse {label}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def authenticate_json(
    path: str | Path,
    *,
    label: str,
    loads: Callable[..., Any] = json.loads,
    allow_resolved_symlink: bool = False,
) -> tuple[AuthenticatedFile, dict[str, Any]]:
    authenticated = AuthenticatedFile.open(
        path,
        label=label,
        allow_resolved_symlink=allow_resolved_symlink,
    )
    try:
        value = parse_json_bytes(authenticated.bytes, label=label, loads=loads)
        authenticated.verify_visible()
        return authenticated, value
    except BaseException:
        authenticated.close()
        raise


def _try_fclonefileat(
    source_descriptor: int,
    destination: Path,
    *,
    parent_descriptor: int | None = None,
) -> bool:
    owned_parent = parent_descriptor is None
    if parent_descriptor is None:
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        parent_descriptor = os.open(destination.parent, directory_flags)
    try:
        try:
            clone = ctypes.CDLL(None, use_errno=True).fclonefileat
        except AttributeError:
            return False
        clone.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        clone.restype = ctypes.c_int
        return clone(
            source_descriptor,
            parent_descriptor,
            os.fsencode(destination.name),
            0,
        ) == 0
    finally:
        if owned_parent:
            os.close(parent_descriptor)


def _visible_identity(
    parent_descriptor: int,
    name: str,
) -> tuple[int, int, int] | None:
    try:
        value = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return None
    return value.st_dev, value.st_ino, value.st_mode


def _unlink_created(
    parent_descriptor: int,
    name: str,
    expected: tuple[int, int, int] | None,
) -> None:
    if expected is None:
        return
    actual = _visible_identity(parent_descriptor, name)
    if actual is None or actual[:2] != expected[:2]:
        return
    os.unlink(name, dir_fd=parent_descriptor)


def clone_or_copy_authenticated(
    source: AuthenticatedFile,
    destination: str | Path,
    *,
    allow_copy_fallback: bool,
    required_free_bytes: int,
) -> None:
    """Create an inode-independent private snapshot from authenticated bytes."""

    target = Path(destination)
    ancestors = _AncestorAuthority.open(target)
    parent = ancestors.parent_descriptor
    created_identity: tuple[int, int, int] | None = None
    try:
        source.verify_visible()
        if _visible_identity(parent, target.name) is not None:
            raise FileExistsError(target)
        cloned = _try_fclonefileat(
            source.descriptor,
            target,
            parent_descriptor=parent,
        )
        if cloned:
            created_identity = _visible_identity(parent, target.name)
            if created_identity is None:
                raise ValueError(f"{source.label} snapshot clone is not visible")
        else:
            if _visible_identity(parent, target.name) is not None:
                raise ValueError(
                    f"{source.label} snapshot clone failed with an unowned destination"
                )
            if not allow_copy_fallback:
                raise ValueError(
                    "copy-on-write recovery snapshot is unavailable and full-copy "
                    "fallback was not explicitly enabled"
                )
            filesystem = os.fstatvfs(parent)
            available = filesystem.f_bavail * filesystem.f_frsize
            if available < required_free_bytes:
                raise ValueError(
                    "recovery snapshot copy fallback requires "
                    f"{required_free_bytes} free bytes, found {available}"
                )
            output = os.open(
                target.name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o400,
                dir_fd=parent,
            )
            opened = os.fstat(output)
            created_identity = (opened.st_dev, opened.st_ino, opened.st_mode)
            copy_primary: BaseException | None = None
            try:
                offset = 0
                while offset < source.size:
                    chunk = os.pread(
                        source.descriptor,
                        min(_READ_CHUNK, source.size - offset),
                        offset,
                    )
                    if not chunk:
                        raise ValueError(
                            f"{source.label} ended while its snapshot was copied"
                        )
                    cursor = 0
                    while cursor < len(chunk):
                        cursor += os.write(output, chunk[cursor:])
                    offset += len(chunk)
                os.fsync(output)
            except BaseException as error:
                copy_primary = error
                raise
            finally:
                try:
                    os.close(output)
                except BaseException:
                    if copy_primary is None:
                        raise
        ancestors.verify()
        linked = os.stat(target.name, dir_fd=parent, follow_symlinks=False)
        if (linked.st_dev, linked.st_ino) == (source.device, source.inode):
            raise ValueError(f"{source.label} snapshot shares its source inode")
        snapshot_descriptor = _open_regular_no_follow(
            target.name,
            label=f"{source.label} snapshot",
            dir_fd=parent,
        )
        try:
            snapshot_identity = FileIdentity.from_stat(os.fstat(snapshot_descriptor))
            snapshot_bytes, snapshot_sha256 = _read_descriptor(snapshot_descriptor)
            if (
                snapshot_identity.size != source.size
                or snapshot_sha256 != source.sha256
                or snapshot_bytes != source.bytes
            ):
                raise ValueError(f"{source.label} snapshot hash drifted")
            os.fchmod(snapshot_descriptor, 0o400)
        finally:
            os.close(snapshot_descriptor)
        source.verify_visible()
        ancestors.verify()
    except BaseException as error:
        try:
            _unlink_created(parent, target.name, created_identity)
        except BaseException:
            pass
        raise
    finally:
        ancestors.close()


@dataclass(slots=True, weakref_slot=True)
class DescriptorSnapshot:
    file_paths: Mapping[str, Path]
    descriptors: tuple[int, ...]
    _path_storage: dict[str, Path] = field(init=False, repr=False)
    _closed: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        storage = dict(self.file_paths)
        self._path_storage = storage
        self.file_paths = MappingProxyType(storage)

    @classmethod
    def create(
        cls,
        sources: Mapping[str, AuthenticatedFile],
        *,
        scratch_dir: str | Path,
        allow_copy_fallback: bool,
        prefix: str = ".authenticated-",
        clone_function: Callable[..., None] | None = None,
        required_free_bytes: int | None = None,
    ) -> "DescriptorSnapshot":
        scratch = Path(scratch_dir)
        temporary = tempfile.TemporaryDirectory(prefix=prefix, dir=scratch)
        root = Path(temporary.name).resolve(strict=True)
        descriptors: list[int] = []
        paths: dict[str, Path] = {}
        remaining = (
            sum(source.size for source in sources.values())
            if required_free_bytes is None
            else required_free_bytes
        )
        try:
            for name, source in sources.items():
                if Path(name).name != name:
                    raise ValueError("descriptor snapshot names must be basenames")
                destination = root / name
                copier = clone_function or clone_or_copy_authenticated
                copier(
                    source,
                    destination,
                    allow_copy_fallback=allow_copy_fallback,
                    required_free_bytes=remaining,
                )
                remaining -= source.size
                retained = AuthenticatedFile.open(
                    destination, label=f"{source.label} snapshot"
                )
                try:
                    if retained.size != source.size or retained.sha256 != source.sha256:
                        raise ValueError(f"{source.label} snapshot hash drifted")
                    descriptor = os.dup(retained.descriptor)
                    descriptors.append(descriptor)
                    _unlink_created(
                        retained._ancestors.parent_descriptor,
                        destination.name,
                        (
                            retained.identity.device,
                            retained.identity.inode,
                            retained.identity.mode,
                        ),
                    )
                    if _visible_identity(
                        retained._ancestors.parent_descriptor, destination.name
                    ) is not None:
                        raise ValueError(
                            f"{source.label} snapshot could not be privately unlinked"
                        )
                finally:
                    retained.close()
                paths[name] = Path(f"/dev/fd/{descriptor}")
            temporary.cleanup()
            return cls(file_paths=paths, descriptors=tuple(descriptors))
        except BaseException:
            for descriptor in descriptors:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            try:
                temporary.cleanup()
            except BaseException:
                pass
            raise

    @classmethod
    def combine(
        cls,
        snapshots: tuple["DescriptorSnapshot", ...],
    ) -> "DescriptorSnapshot":
        """Transfer live descriptors from component snapshots into one owner."""

        paths: dict[str, Path] = {}
        descriptors: list[int] = []
        for snapshot in snapshots:
            if snapshot._closed:
                raise ValueError("cannot combine a closed descriptor snapshot")
            overlap = set(paths).intersection(snapshot.file_paths)
            if overlap:
                raise ValueError("descriptor snapshot names must be unique")
            paths.update(snapshot.file_paths)
            descriptors.extend(snapshot.descriptors)
        for snapshot in snapshots:
            snapshot._path_storage.clear()
            snapshot.descriptors = ()
            snapshot._closed = True
        return cls(paths, tuple(descriptors))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        descriptors = self.descriptors
        self.descriptors = ()
        self._path_storage.clear()
        for descriptor in descriptors:
            try:
                os.close(descriptor)
            except OSError:
                pass

    def __enter__(self) -> "DescriptorSnapshot":
        if self._closed:
            raise ValueError("descriptor snapshot is closed")
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except AttributeError:
            pass


@dataclass(frozen=True, slots=True)
class PublishedArtifact:
    path: Path
    sha256: str
    size: int
    identity: FileIdentity
    displaced_path: Path | None = None
    displaced_identity: FileIdentity | None = None
    displaced_sha256: str | None = None


def _before_publication_syscall(
    _target: Path,
    _parent_descriptor: int,
    _staging_name: str,
) -> None:
    """Test seam immediately before the publication pathname syscall."""


def _after_publication_syscall(
    _target: Path,
    _parent_descriptor: int,
    _staging_name: str,
) -> None:
    """Test seam immediately after the publication pathname syscall."""


def _before_replacement_exchange(
    _target: Path,
    _parent_descriptor: int,
    _staging_name: str,
) -> None:
    """Test seam after destination auth and immediately before exchange."""


def _before_displaced_evidence_return(
    _target: Path,
    _parent_descriptor: int,
    _staging_name: str,
) -> None:
    """Test seam before final reauthentication of displaced evidence."""


class _AtomicExchangeUnavailable(ValueError):
    pass


def _exchange_paths(
    parent_descriptor: int,
    first: str,
    second: str,
) -> None:
    """Atomically exchange two existing names or fail before ordinary rename."""

    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        try:
            exchange = libc.renameatx_np
        except AttributeError as error:
            raise _AtomicExchangeUnavailable(
                "atomic replacement exchange is unavailable on this host"
            ) from error
        exchange.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        exchange.restype = ctypes.c_int
        result = exchange(
            parent_descriptor,
            os.fsencode(first),
            parent_descriptor,
            os.fsencode(second),
            0x00000002,  # RENAME_SWAP
        )
    elif sys.platform.startswith("linux"):
        try:
            exchange = libc.renameat2
        except AttributeError as error:
            raise _AtomicExchangeUnavailable(
                "atomic replacement exchange is unavailable on this host"
            ) from error
        exchange.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        exchange.restype = ctypes.c_int
        result = exchange(
            parent_descriptor,
            os.fsencode(first),
            parent_descriptor,
            os.fsencode(second),
            0x00000002,  # RENAME_EXCHANGE
        )
    else:
        raise _AtomicExchangeUnavailable(
            "atomic replacement exchange is unavailable on this host"
        )
    if result != 0:
        error_number = ctypes.get_errno() or errno.EIO
        raise OSError(error_number, os.strerror(error_number))


def _move_noreplace(
    source_parent: int,
    source_name: str,
    destination_parent: int,
    destination_name: str,
) -> None:
    """Move one name without ever replacing a destination entry."""

    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        try:
            move = libc.renameatx_np
        except AttributeError as error:
            raise _AtomicExchangeUnavailable(
                "exclusive forensic move is unavailable on this host"
            ) from error
        flag = 0x00000004  # RENAME_EXCL
    elif sys.platform.startswith("linux"):
        try:
            move = libc.renameat2
        except AttributeError as error:
            raise _AtomicExchangeUnavailable(
                "exclusive forensic move is unavailable on this host"
            ) from error
        flag = 0x00000001  # RENAME_NOREPLACE
    else:
        raise _AtomicExchangeUnavailable(
            "exclusive forensic move is unavailable on this host"
        )
    move.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    move.restype = ctypes.c_int
    result = move(
        source_parent,
        os.fsencode(source_name),
        destination_parent,
        os.fsencode(destination_name),
        flag,
    )
    if result != 0:
        error_number = ctypes.get_errno() or errno.EIO
        raise OSError(error_number, os.strerror(error_number))


def _reconcile_publication_failure(
    *,
    target: Path,
    ancestors: _AncestorAuthority,
    parent_descriptor: int,
    expected_identity: tuple[int, int],
    expected_payload: bytes,
    expected_sha256: str,
    failure: BaseException,
) -> PublicationCorruptError | PublicationUncertainError:
    """Classify a possible commit without mutating either visible pathname."""

    if isinstance(failure, PublicationCorruptError):
        return failure
    try:
        ancestors.verify()
        descriptor = _open_regular_no_follow(
            target.name,
            label="published artifact",
            dir_fd=parent_descriptor,
        )
        try:
            identity = os.fstat(descriptor)
            payload, digest = _read_descriptor(descriptor)
        finally:
            os.close(descriptor)
    except BaseException as authentication_error:
        return PublicationCorruptError(
            f"published artifact could not be authenticated after publication: {target}"
        )
    if (
        (identity.st_dev, identity.st_ino) != expected_identity
        or payload != expected_payload
        or digest != expected_sha256
    ):
        return PublicationCorruptError(
            f"published artifact is corrupt after publication: {target}"
        )
    if isinstance(failure, PublicationUncertainError):
        return failure
    return PublicationUncertainError(
        f"published artifact committed but completion is uncertain: {target}"
    )


def publish_file_transactionally(
    source: str | Path,
    destination: str | Path,
    *,
    allow_replace: bool = False,
    expected_destination: ReplacementAuthority | None = None,
    mode: int = 0o400,
    label: str = "publication source",
) -> PublishedArtifact:
    """Authenticate one staged file and publish precisely those bytes."""

    authenticated = AuthenticatedFile.open(source, label=label)
    primary: BaseException | None = None
    committed = False
    try:
        authenticated.verify_visible()
        published = publish_bytes_transactionally(
            destination,
            authenticated.bytes,
            allow_replace=allow_replace,
            expected_destination=expected_destination,
            before_publication=authenticated.verify_visible,
            mode=mode,
        )
        committed = True
        try:
            authenticated.verify_visible()
        except BaseException as error:
            raise PublicationCorruptError(
                f"publication source changed after destination commit: {destination}"
            ) from error
        return published
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            authenticated.close()
        except BaseException as error:
            if primary is None:
                if committed:
                    raise PublicationUncertainError(
                        "published artifact committed but source descriptor cleanup "
                        f"is uncertain: {destination}"
                    ) from error
                raise


def publish_bytes_transactionally(
    destination: str | Path,
    payload: bytes,
    *,
    allow_replace: bool = False,
    expected_destination: ReplacementAuthority | None = None,
    before_publication: Callable[[], None] | None = None,
    mode: int = 0o400,
) -> PublishedArtifact:
    """Publish exact bytes through one parent descriptor and verify the result."""

    target = Path(destination)
    if allow_replace and expected_destination is None:
        raise ValueError(
            "replacement authority requires the expected authenticated destination identity"
        )
    if expected_destination is not None and not isinstance(
        expected_destination, ReplacementAuthority
    ):
        raise ValueError("replacement authority must be authenticated")
    replacing = expected_destination is not None
    ancestors = _AncestorAuthority.open(target)
    parent = ancestors.parent_descriptor
    staging_name = f".{target.name}.partial-{next(tempfile._get_candidate_names())}"
    staging = -1
    staging_identity: tuple[int, int, int] | None = None
    published = False
    publication_attempted = False
    primary: BaseException | None = None
    retained_displaced_path: Path | None = None
    retained_displaced_identity: FileIdentity | None = None
    retained_displaced_sha256: str | None = None
    try:
        staging = os.open(
            staging_name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            mode,
            dir_fd=parent,
        )
        opened_staging = os.fstat(staging)
        staging_identity = (
            opened_staging.st_dev,
            opened_staging.st_ino,
            opened_staging.st_mode,
        )
        cursor = 0
        while cursor < len(payload):
            cursor += os.write(staging, payload[cursor:])
        os.fsync(staging)
        ancestors.verify()
        visible_source = _open_regular_no_follow(
            staging_name,
            label="publication source",
            dir_fd=parent,
        )
        try:
            visible_source_identity = FileIdentity.from_stat(os.fstat(visible_source))
            source_bytes, source_digest = _read_descriptor(visible_source)
            if (
                source_bytes != payload
                or (visible_source_identity.device, visible_source_identity.inode)
                != (opened_staging.st_dev, opened_staging.st_ino)
            ):
                raise ValueError(f"publication source bytes changed: {target}")
        finally:
            os.close(visible_source)
        descriptor_bytes, descriptor_digest = _read_descriptor(staging)
        if descriptor_bytes != payload or descriptor_digest != source_digest:
            raise ValueError(f"publication source descriptor changed: {target}")
        ancestors.verify()
        _before_publication_syscall(target, parent, staging_name)
        ancestors.verify()
        if _visible_identity(parent, staging_name) != staging_identity:
            raise ValueError(f"publication source was substituted: {target}")
        if before_publication is not None:
            before_publication()
        immediate_bytes, immediate_digest = _read_descriptor(staging)
        immediate_identity = os.fstat(staging)
        if (
            immediate_bytes != payload
            or immediate_digest != source_digest
            or (immediate_identity.st_dev, immediate_identity.st_ino)
            != (opened_staging.st_dev, opened_staging.st_ino)
        ):
            raise ValueError(f"publication source changed before commit: {target}")
        if replacing:
            try:
                destination_descriptor = _open_regular_no_follow(
                    target.name,
                    label="replacement destination",
                    dir_fd=parent,
                )
                try:
                    actual_destination = FileIdentity.from_stat(
                        os.fstat(destination_descriptor)
                    )
                    _destination_payload, destination_sha256 = _read_descriptor(
                        destination_descriptor
                    )
                finally:
                    os.close(destination_descriptor)
            except BaseException as error:
                raise ValueError(
                    f"replacement destination identity could not be authenticated: {target}"
                ) from error
            assert expected_destination is not None
            if (
                actual_destination != expected_destination.identity
                or destination_sha256 != expected_destination.sha256
            ):
                raise ValueError(
                    f"replacement destination identity changed before publication: {target}"
                )
            _before_replacement_exchange(target, parent, staging_name)
        if before_publication is not None:
            before_publication()
        if replacing:
            publication_attempted = True
            try:
                _exchange_paths(parent, staging_name, target.name)
            except _AtomicExchangeUnavailable:
                publication_attempted = False
                raise
        else:
            publication_attempted = True
            os.link(
                staging_name,
                target.name,
                src_dir_fd=parent,
                dst_dir_fd=parent,
                follow_symlinks=False,
            )
        published = True
        _after_publication_syscall(target, parent, staging_name)
        try:
            descriptor = os.open(
                target.name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent,
            )
            try:
                identity = FileIdentity.from_stat(os.fstat(descriptor))
                actual, digest = _read_descriptor(descriptor)
                if (
                    actual != payload
                    or digest != source_digest
                    or (identity.device, identity.inode)
                    != (opened_staging.st_dev, opened_staging.st_ino)
                ):
                    raise PublicationCorruptError(
                        f"published artifact bytes do not match source: {target}"
                    )
            finally:
                os.close(descriptor)
        except PublicationCorruptError:
            raise
        except OSError as error:
            raise PublicationCorruptError(
                f"published artifact could not be authenticated: {target}"
            ) from error
        if replacing:
            assert expected_destination is not None
            try:
                displaced_descriptor = _open_regular_no_follow(
                    staging_name,
                    label="displaced replacement destination",
                    dir_fd=parent,
                )
                try:
                    displaced_identity = FileIdentity.from_stat(
                        os.fstat(displaced_descriptor)
                    )
                    _displaced_payload, displaced_sha256 = _read_descriptor(
                        displaced_descriptor
                    )
                finally:
                    os.close(displaced_descriptor)
            except BaseException as error:
                if isinstance(error, PublicationCorruptError):
                    raise
                raise PublicationCorruptError(
                    f"displaced replacement could not be authenticated: {target}"
                ) from error
            expected_identity = expected_destination.identity
            stable_displaced_identity = (
                displaced_identity.device,
                displaced_identity.inode,
                displaced_identity.mode,
                displaced_identity.size,
                displaced_identity.mtime_ns,
                displaced_identity.link_count,
            )
            stable_expected_identity = (
                expected_identity.device,
                expected_identity.inode,
                expected_identity.mode,
                expected_identity.size,
                expected_identity.mtime_ns,
                expected_identity.link_count,
            )
            if (
                stable_displaced_identity != stable_expected_identity
                or displaced_identity.ctime_ns < expected_identity.ctime_ns
                or displaced_sha256 != expected_destination.sha256
            ):
                raise PublicationCorruptError(
                    f"displaced replacement is corrupt after exchange: {target}"
                )
            _before_displaced_evidence_return(target, parent, staging_name)
            try:
                final_displaced_descriptor = _open_regular_no_follow(
                    staging_name,
                    label="displaced replacement destination",
                    dir_fd=parent,
                )
                try:
                    final_displaced_identity = FileIdentity.from_stat(
                        os.fstat(final_displaced_descriptor)
                    )
                    _final_displaced_payload, final_displaced_sha256 = _read_descriptor(
                        final_displaced_descriptor
                    )
                finally:
                    os.close(final_displaced_descriptor)
            except BaseException as error:
                raise PublicationCorruptError(
                    f"displaced replacement changed before cleanup: {target}"
                ) from error
            final_stable_identity = (
                final_displaced_identity.device,
                final_displaced_identity.inode,
                final_displaced_identity.mode,
                final_displaced_identity.size,
                final_displaced_identity.mtime_ns,
                final_displaced_identity.link_count,
            )
            if (
                final_stable_identity != stable_expected_identity
                or final_displaced_identity.ctime_ns < expected_identity.ctime_ns
                or final_displaced_sha256 != expected_destination.sha256
            ):
                raise PublicationCorruptError(
                    f"displaced replacement changed before cleanup: {target}"
                )
            retained_displaced_path = target.parent / staging_name
            retained_displaced_identity = final_displaced_identity
            retained_displaced_sha256 = final_displaced_sha256
        else:
            _unlink_created(parent, staging_name, staging_identity)
        ancestors.verify()
        try:
            os.fsync(parent)
        except OSError as error:
            raise PublicationUncertainError(
                f"published artifact durability is uncertain: {target}"
            ) from error
        ancestors.verify()
        try:
            final_descriptor = _open_regular_no_follow(
                target.name,
                label="published artifact",
                dir_fd=parent,
            )
            try:
                final_identity = FileIdentity.from_stat(os.fstat(final_descriptor))
                final_payload, final_digest = _read_descriptor(final_descriptor)
            finally:
                os.close(final_descriptor)
        except BaseException as error:
            if isinstance(error, PublicationCorruptError):
                raise
            raise PublicationCorruptError(
                f"published artifact could not be authenticated after durability: {target}"
            ) from error
        if (
            (final_identity.device, final_identity.inode)
            != (opened_staging.st_dev, opened_staging.st_ino)
            or final_payload != payload
            or final_digest != source_digest
        ):
            raise PublicationCorruptError(
                f"published artifact is corrupt after durability: {target}"
            )
        os.close(staging)
        staging = -1
        return PublishedArtifact(
            target,
            digest,
            len(payload),
            identity,
            displaced_path=retained_displaced_path,
            displaced_identity=retained_displaced_identity,
            displaced_sha256=retained_displaced_sha256,
        )
    except BaseException as error:
        if publication_attempted and not (
            isinstance(error, FileExistsError) and not replacing and not published
        ):
            classified = _reconcile_publication_failure(
                target=target,
                ancestors=ancestors,
                parent_descriptor=parent,
                expected_identity=(opened_staging.st_dev, opened_staging.st_ino),
                expected_payload=payload,
                expected_sha256=source_digest,
                failure=error,
            )
            primary = classified
            raise classified from error
        primary = error
        raise
    finally:
        if staging >= 0:
            try:
                os.close(staging)
            except OSError:
                pass
        if not published and not (
            publication_attempted and isinstance(primary, PublicationCorruptError)
        ):
            try:
                _unlink_created(parent, staging_name, staging_identity)
            except OSError:
                if primary is None:
                    raise
        elif not replacing and not isinstance(primary, PublicationCorruptError):
            try:
                _unlink_created(parent, staging_name, staging_identity)
            except OSError:
                if primary is None:
                    raise
        ancestors.close()


def archive_displaced_artifact(
    published: PublishedArtifact,
    archive_directory: str | Path,
) -> Path | None:
    """Explicitly move retained displaced evidence; never delete it."""

    if published.displaced_path is None:
        return None
    if (
        published.displaced_identity is None
        or published.displaced_sha256 is None
    ):
        raise ValueError("displaced publication evidence is incomplete")
    displaced = AuthenticatedFile.open(
        published.displaced_path,
        label="displaced publication evidence",
    )
    archive_path = Path(archive_directory) / (
        f"{published.path.name}-{published.displaced_path.name.lstrip('.')}"
    )
    archive_authority = _AncestorAuthority.open(archive_path)
    moved = False
    try:
        if (
            displaced.identity != published.displaced_identity
            or displaced.sha256 != published.displaced_sha256
        ):
            raise PublicationCorruptError(
                f"displaced publication evidence changed before archival: {displaced.path}"
            )
        if _visible_identity(
            archive_authority.parent_descriptor, archive_path.name
        ) is not None:
            raise FileExistsError(archive_path)
        _move_noreplace(
            displaced._ancestors.parent_descriptor,
            displaced._opened_path.name,
            archive_authority.parent_descriptor,
            archive_path.name,
        )
        moved = True
        archived = AuthenticatedFile.open(
            archive_path,
            label="archived displaced publication evidence",
        )
        try:
            expected = published.displaced_identity
            stable_archived = (
                archived.identity.device,
                archived.identity.inode,
                archived.identity.mode,
                archived.identity.size,
                archived.identity.mtime_ns,
                archived.identity.link_count,
            )
            stable_expected = (
                expected.device,
                expected.inode,
                expected.mode,
                expected.size,
                expected.mtime_ns,
                expected.link_count,
            )
            if (
                stable_archived != stable_expected
                or archived.identity.ctime_ns < expected.ctime_ns
                or archived.sha256 != published.displaced_sha256
            ):
                raise PublicationCorruptError(
                    f"archived displaced evidence is corrupt: {archive_path}"
                )
        finally:
            archived.close()
        if _visible_identity(
            displaced._ancestors.parent_descriptor,
            displaced._opened_path.name,
        ) is not None:
            raise PublicationCorruptError(
                f"displaced evidence remained at its publication path: {displaced.path}"
            )
        os.fsync(displaced._ancestors.parent_descriptor)
        os.fsync(archive_authority.parent_descriptor)
        return archive_path
    except (PublicationCorruptError, PublicationUncertainError):
        raise
    except BaseException as error:
        if moved:
            raise PublicationUncertainError(
                f"displaced evidence archival is uncertain: {archive_path}"
            ) from error
        raise
    finally:
        archive_authority.close()
        displaced.close()


__all__ = [
    "AuthenticatedFile",
    "archive_displaced_artifact",
    "DescriptorSnapshot",
    "FileIdentity",
    "PublicationCorruptError",
    "PublicationUncertainError",
    "PublishedArtifact",
    "ReplacementAuthority",
    "authenticate_json",
    "clone_or_copy_authenticated",
    "open_regular_no_follow",
    "parse_json_bytes",
    "publish_bytes_transactionally",
    "publish_file_transactionally",
]
