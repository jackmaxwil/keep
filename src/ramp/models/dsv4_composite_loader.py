"""Authenticated composite loading for the Task-4 DeepSeek-V4 artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from keep.io.source_safetensors import read_safetensors_file_header
from ramp.models.deepseek_v4_flash_adapter import (
    DeepseekV4FlashNonVQBindReport,
    DeepseekV4FlashVQModel,
    bind_deepseek_v4_flash_mtp_vq_experts,
    bind_deepseek_v4_flash_non_vq_weights,
    bind_deepseek_v4_flash_vq_experts,
    deepseek_v4_flash_args_from_config,
    dense_deepseek_v4_flash_routed_parameter_names,
    has_unbound_deepseek_v4_flash_mtp_experts,
    has_unbound_deepseek_v4_flash_vq_experts,
)

MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash-0731"
REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
CONFIG_SHA256 = "6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023"
INDEX_SHA256 = "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"
RESIDENT_MANIFEST_SHA256 = (
    "35fa6361691326aa805a5ff59a95508af1740660b98844071da59856ae8009d5"
)
RESIDENT_PACKAGE_SET_SHA256 = (
    "75b63dee580fc4038be088db63ef9e5dd2cb0256bb722abb77d94651ee51abe9"
)
VQ_MANIFEST_SHA256 = "a996c4bedc514f49c54fe15d5d9b2373f839678e55b241861f2fda60cb4bf8b3"
VQ_INVENTORY_SHA256 = "4bda0cf1a635123d08f711cd38ad4bb3f716b41714883495ec91cfd71540e843"
_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class Dsv4VqManifestInventory:
    backbone_indices: tuple[int, ...]
    mtp_indices: tuple[int, ...]
    file_records: Mapping[str, Mapping[str, Any]]
    inventory_sha256: str


@dataclass(frozen=True)
class Dsv4ValidatedCompositeInputs:
    source_dir: Path
    resident_dir: Path
    vq_dir: Path
    config: Mapping[str, Any]
    resident_manifest: Mapping[str, Any]
    vq_manifest: Mapping[str, Any]
    vq_inventory: Dsv4VqManifestInventory
    artifact_paths: Mapping[str, Path]
    payload_receipt: Mapping[str, Any]
    payload_receipt_sha256: str
    identity: Mapping[str, Any]
    identity_sha256: str


@dataclass(frozen=True)
class Dsv4CompositeLoadReport:
    artifact_identity_sha256: str
    non_vq_bind_report: DeepseekV4FlashNonVQBindReport
    bound_backbone_layers: tuple[int, ...]
    bound_mtp_stages: tuple[int, ...]
    dense_routed_parameter_names: tuple[str, ...]
    unbound_backbone_vq_experts: bool
    unbound_mtp_vq_experts: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_identity_sha256": self.artifact_identity_sha256,
            "non_vq_bind_report": self.non_vq_bind_report.to_dict(),
            "bound_backbone_layers": list(self.bound_backbone_layers),
            "bound_mtp_stages": list(self.bound_mtp_stages),
            "dense_routed_parameter_names": list(self.dense_routed_parameter_names),
            "unbound_backbone_vq_experts": self.unbound_backbone_vq_experts,
            "unbound_mtp_vq_experts": self.unbound_mtp_vq_experts,
        }


@dataclass(frozen=True)
class Dsv4LoadedComposite:
    model: DeepseekV4FlashVQModel
    inputs: Dsv4ValidatedCompositeInputs
    report: Dsv4CompositeLoadReport


def validate_dsv4_vq_manifest_structure(
    manifest: Mapping[str, Any],
    *,
    expected_backbone_layers: int = 43,
    expected_mtp_blocks: int = 3,
) -> Dsv4VqManifestInventory:
    """Partition the routed manifest into one exact file per block/projection."""

    if manifest.get("record_type") != "dsv4_vq_artifact_manifest_v1":
        raise ValueError("invalid DSV4 VQ manifest record type")
    if manifest.get("family") != "deepseek-v4-flash":
        raise ValueError("invalid DSV4 VQ artifact family")
    source = manifest.get("source")
    expected_source = {
        "hf_model_id": MODEL_ID,
        "revision": REVISION,
        "config_sha256": CONFIG_SHA256,
        "index_sha256": INDEX_SHA256,
    }
    if not isinstance(source, Mapping) or any(
        source.get(key) != value for key, value in expected_source.items()
    ):
        raise ValueError("DSV4 VQ source identity does not match the pinned release")
    geometry = manifest.get("geometry")
    if not isinstance(geometry, Mapping) or (
        geometry.get("backbone_layers") != expected_backbone_layers
        or geometry.get("mtp_blocks") != expected_mtp_blocks
        or geometry.get("moe_blocks") != expected_backbone_layers + expected_mtp_blocks
    ):
        raise ValueError("DSV4 VQ geometry does not match the exact runtime inventory")

    blocks = manifest.get("blocks")
    if not isinstance(blocks, list):
        raise TypeError("DSV4 VQ block inventory must be an array")
    expected_indices = {
        "backbone": tuple(range(expected_backbone_layers)),
        "mtp": tuple(range(expected_mtp_blocks)),
    }
    seen_blocks: dict[str, list[int]] = {"backbone": [], "mtp": []}
    files: dict[str, Mapping[str, Any]] = {}
    inventory_rows: list[tuple[str, int, str]] = []
    for block in blocks:
        if not isinstance(block, Mapping):
            raise TypeError("DSV4 VQ block inventory contains a non-object")
        kind = str(block.get("kind"))
        if kind not in seen_blocks:
            raise ValueError(f"DSV4 VQ block has invalid kind {kind!r}")
        index = block.get("index")
        if type(index) is not int:
            raise ValueError("DSV4 VQ block index must be an integer")
        if index in seen_blocks[kind]:
            raise ValueError(f"duplicate DSV4 VQ {kind} block {index}")
        seen_blocks[kind].append(index)
        stem = f"layer-{index:05d}" if kind == "backbone" else f"mtp-{index:05d}"
        key = f"layers.{index}" if kind == "backbone" else f"mtp.{index}"
        prefix = (
            f"model.layers.{index}.ffn.switch_mlp"
            if kind == "backbone"
            else f"mtp_drafter.blocks.{index}.ffn.switch_mlp"
        )
        if (
            block.get("file_stem") != stem
            or block.get("key") != key
            or block.get("tensor_prefix") != prefix
        ):
            raise ValueError(f"DSV4 VQ {key} discovery metadata is invalid")
        records = block.get("files")
        if not isinstance(records, list) or len(records) != len(_PROJECTIONS):
            raise ValueError(f"DSV4 VQ {key} file inventory is not exact")
        seen_projections: set[str] = set()
        for record in records:
            if not isinstance(record, Mapping):
                raise TypeError(f"DSV4 VQ {key} file inventory is invalid")
            projection = str(record.get("projection"))
            filename = f"{stem}-{projection}.safetensors"
            digest = str(record.get("sha256", ""))
            size = record.get("bytes")
            if (
                projection not in _PROJECTIONS
                or projection in seen_projections
                or record.get("path") != filename
                or type(size) is not int
                or size <= 0
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"DSV4 VQ {key} file inventory is invalid")
            if filename in files:
                raise ValueError(f"duplicate DSV4 VQ artifact file {filename}")
            seen_projections.add(projection)
            files[filename] = record
            inventory_rows.append((filename, size, digest))
        if seen_projections != set(_PROJECTIONS):
            raise ValueError(f"DSV4 VQ {key} projection inventory is not exact")

    backbone = tuple(sorted(seen_blocks["backbone"]))
    mtp = tuple(sorted(seen_blocks["mtp"]))
    if backbone != expected_indices["backbone"] or mtp != expected_indices["mtp"]:
        raise ValueError(
            "DSV4 VQ manifest requires the exact backbone and MTP inventory"
        )
    return Dsv4VqManifestInventory(
        backbone_indices=backbone,
        mtp_indices=mtp,
        file_records=dict(sorted(files.items())),
        inventory_sha256=_canonical_sha256(sorted(inventory_rows)),
    )


def _file_identity(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
    }


def _tensor_inventory(path: Path) -> tuple[dict[str, dict[str, Any]], str]:
    header = read_safetensors_file_header(path)
    inventory = {
        name: {"dtype": tensor.dtype, "shape": list(tensor.shape)}
        for name, tensor in sorted(header.tensors.items())
    }
    return inventory, _canonical_sha256(inventory)


def _payload_declarations(
    *,
    resident_dir: Path,
    resident_manifest: Mapping[str, Any],
    vq_dir: Path,
    vq_inventory: Dsv4VqManifestInventory,
) -> dict[tuple[str, str], dict[str, Any]]:
    declarations: dict[tuple[str, str], dict[str, Any]] = {}

    shards = resident_manifest.get("shards")
    tensors = resident_manifest.get("tensors")
    if not isinstance(shards, list) or not isinstance(tensors, list):
        raise TypeError("DSV4 resident payload inventory is missing")
    index = json.loads((resident_dir / "model.safetensors.index.json").read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping):
        raise TypeError("DSV4 resident package index weight_map is missing")
    tensor_records: dict[str, Mapping[str, Any]] = {}
    manifest_weight_map: dict[str, str] = {}
    for record in tensors:
        if not isinstance(record, Mapping):
            raise TypeError("DSV4 resident tensor inventory is invalid")
        name = str(record.get("name"))
        shard = str(record.get("output_shard"))
        if name in tensor_records:
            raise ValueError(f"duplicate DSV4 resident tensor {name}")
        tensor_records[name] = record
        manifest_weight_map[name] = shard
    if dict(weight_map) != manifest_weight_map:
        raise ValueError("DSV4 resident index and manifest tensor inventory differ")

    expected_resident_files: set[str] = set()
    for record in shards:
        if not isinstance(record, Mapping):
            raise TypeError("DSV4 resident shard inventory is invalid")
        filename = str(record.get("filename"))
        key = ("resident", filename)
        if key in declarations:
            raise ValueError(f"duplicate DSV4 resident shard {filename}")
        expected_resident_files.add(filename)
        expected_tensors = {
            name for name, shard in manifest_weight_map.items() if shard == filename
        }
        if int(record.get("tensor_count", -1)) != len(expected_tensors):
            raise ValueError(f"DSV4 resident shard tensor count differs: {filename}")
        declarations[key] = {
            "path": resident_dir / filename,
            "declared_sha256": str(record.get("file_sha256", "")),
            "file_bytes": record.get("file_bytes"),
            "expected_tensors": expected_tensors,
        }
    actual_resident_files = {
        str(path.relative_to(resident_dir))
        for path in resident_dir.rglob("*.safetensors")
    }
    if actual_resident_files != expected_resident_files:
        raise ValueError("DSV4 resident runtime safetensors inventory is not exact")

    expected_vq_files = set(vq_inventory.file_records)
    actual_vq_files = {
        str(path.relative_to(vq_dir)) for path in vq_dir.rglob("*.safetensors")
    }
    if actual_vq_files != expected_vq_files:
        raise ValueError("DSV4 routed runtime safetensors inventory is not exact")
    for filename, record in vq_inventory.file_records.items():
        codes = record.get("codes_tensor")
        scales = record.get("scales_tensor")
        if not isinstance(codes, str) or not isinstance(scales, str):
            raise TypeError(f"DSV4 routed tensor discovery is invalid: {filename}")
        declarations[("vq", filename)] = {
            "path": vq_dir / filename,
            "declared_sha256": str(record.get("sha256", "")),
            "file_bytes": record.get("bytes"),
            "expected_tensors": {"model.vq_codebook.e8", codes, scales},
        }
    return declarations


def _validate_declared_payload(
    component: str,
    relative_path: str,
    declaration: Mapping[str, Any],
) -> tuple[Path, dict[str, dict[str, Any]], str]:
    path = Path(declaration["path"])
    expected_size = declaration.get("file_bytes")
    digest = str(declaration.get("declared_sha256", ""))
    if (
        path.is_symlink()
        or not path.is_file()
        or type(expected_size) is not int
        or path.stat().st_size != expected_size
        or len(digest) != 64
    ):
        raise ValueError(
            f"DSV4 {component} payload declaration is invalid: {relative_path}"
        )
    inventory, inventory_sha256 = _tensor_inventory(path)
    if set(inventory) != set(declaration["expected_tensors"]):
        raise ValueError(
            f"DSV4 {component} tensor inventory is not exact: {relative_path}"
        )
    return path, inventory, inventory_sha256


def authenticate_dsv4_payloads(
    *,
    resident_dir: str | Path,
    resident_manifest: Mapping[str, Any],
    vq_dir: str | Path,
    vq_inventory: Dsv4VqManifestInventory,
) -> dict[str, Any]:
    """Hash and inventory every resident/VQ payload before the first bind."""

    resident_root = Path(resident_dir).resolve()
    vq_root = Path(vq_dir).resolve()
    declarations = _payload_declarations(
        resident_dir=resident_root,
        resident_manifest=resident_manifest,
        vq_dir=vq_root,
        vq_inventory=vq_inventory,
    )
    files: list[dict[str, Any]] = []
    for (component, relative_path), declaration in sorted(declarations.items()):
        path, inventory, inventory_sha256 = _validate_declared_payload(
            component, relative_path, declaration
        )
        before = _file_identity(path)
        content_sha256 = _sha256(path)
        after = _file_identity(path)
        if before != after:
            raise ValueError(f"DSV4 payload changed while hashing: {relative_path}")
        if content_sha256 != declaration["declared_sha256"]:
            raise ValueError(f"DSV4 payload content SHA-256 mismatch: {relative_path}")
        files.append(
            {
                "component": component,
                "relative_path": relative_path,
                "declared_sha256": declaration["declared_sha256"],
                "content_sha256": content_sha256,
                "file_bytes": after["size"],
                "current_file_identity": after,
                "tensor_count": len(inventory),
                "tensor_inventory_sha256": inventory_sha256,
            }
        )
    return {
        "record_type": "dsv4_payload_content_receipt_v1",
        "schema_version": 1,
        "resident_dir": str(resident_root),
        "vq_dir": str(vq_root),
        "files": files,
    }


def dsv4_payload_receipt_sha256(receipt: Mapping[str, Any]) -> str:
    return _canonical_sha256(receipt)


def validate_dsv4_payload_receipt(
    *,
    resident_dir: str | Path,
    resident_manifest: Mapping[str, Any],
    vq_dir: str | Path,
    vq_inventory: Dsv4VqManifestInventory,
    receipt: Mapping[str, Any],
    expected_receipt_sha256: str,
) -> str:
    """Seal a prior content hash to unchanged robust file/header identities."""

    digest = dsv4_payload_receipt_sha256(receipt)
    if digest != expected_receipt_sha256:
        raise ValueError("DSV4 payload receipt SHA-256 mismatch")
    resident_root = Path(resident_dir).resolve()
    vq_root = Path(vq_dir).resolve()
    if (
        receipt.get("record_type") != "dsv4_payload_content_receipt_v1"
        or receipt.get("resident_dir") != str(resident_root)
        or receipt.get("vq_dir") != str(vq_root)
    ):
        raise ValueError("DSV4 payload receipt authority is invalid")
    declarations = _payload_declarations(
        resident_dir=resident_root,
        resident_manifest=resident_manifest,
        vq_dir=vq_root,
        vq_inventory=vq_inventory,
    )
    records = receipt.get("files")
    if not isinstance(records, list):
        raise TypeError("DSV4 payload receipt file inventory is missing")
    receipt_files: dict[tuple[str, str], Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("DSV4 payload receipt file record is invalid")
        key = (str(record.get("component")), str(record.get("relative_path")))
        if key in receipt_files:
            raise ValueError(f"duplicate DSV4 payload receipt file: {key}")
        receipt_files[key] = record
    if set(receipt_files) != set(declarations):
        raise ValueError("DSV4 payload receipt file inventory is not exact")

    for (component, relative_path), declaration in declarations.items():
        record = receipt_files[(component, relative_path)]
        if (
            record.get("declared_sha256") != declaration["declared_sha256"]
            or record.get("content_sha256") != declaration["declared_sha256"]
            or record.get("file_bytes") != declaration["file_bytes"]
        ):
            raise ValueError(
                f"DSV4 payload receipt declaration drifted: {relative_path}"
            )
        path, inventory, inventory_sha256 = _validate_declared_payload(
            component, relative_path, declaration
        )
        if record.get("current_file_identity") != _file_identity(path):
            raise ValueError(
                f"DSV4 payload current file identity changed: {relative_path}"
            )
        if (
            record.get("tensor_count") != len(inventory)
            or record.get("tensor_inventory_sha256") != inventory_sha256
        ):
            raise ValueError(f"DSV4 payload tensor inventory changed: {relative_path}")
    return digest


def validate_dsv4_artifact_identity(identity: Mapping[str, Any]) -> str:
    """Recompute the exact pinned, content-bound composite identity."""

    expected = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "config_sha256": CONFIG_SHA256,
        "index_sha256": INDEX_SHA256,
        "resident_manifest_sha256": RESIDENT_MANIFEST_SHA256,
        "resident_package_set_sha256": RESIDENT_PACKAGE_SET_SHA256,
        "resident_parameter_tensors": 1271,
        "vq_manifest_sha256": VQ_MANIFEST_SHA256,
        "vq_inventory_sha256": VQ_INVENTORY_SHA256,
        "vq_projection_files": 138,
    }
    if not isinstance(identity, Mapping):
        raise TypeError("DSV4 artifact identity mapping is missing")
    if set(identity) != {*expected, "payload_content_receipt_sha256"} or any(
        identity.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("DSV4 artifact identity does not match pinned authority")
    receipt_sha256 = identity.get("payload_content_receipt_sha256")
    if (
        not isinstance(receipt_sha256, str)
        or len(receipt_sha256) != 64
        or any(character not in "0123456789abcdef" for character in receipt_sha256)
    ):
        raise ValueError("DSV4 artifact identity payload receipt SHA-256 is invalid")
    return _canonical_sha256(identity)


def validate_dsv4_composite_inputs(
    *,
    source_dir: str | Path,
    resident_dir: str | Path,
    vq_dir: str | Path,
    expected_resident_manifest_sha256: str = RESIDENT_MANIFEST_SHA256,
    expected_vq_manifest_sha256: str = VQ_MANIFEST_SHA256,
    authenticate_payloads: bool = False,
    payload_receipt: Mapping[str, Any] | None = None,
    expected_payload_receipt_sha256: str | None = None,
) -> Dsv4ValidatedCompositeInputs:
    """Authenticate source authority and every artifact path before binding."""

    source_root = Path(source_dir)
    resident_root = Path(resident_dir)
    vq_root = Path(vq_dir)
    config_path = source_root / "config.json"
    index_path = source_root / "model.safetensors.index.json"
    if _sha256(config_path) != CONFIG_SHA256 or _sha256(index_path) != INDEX_SHA256:
        raise ValueError("DSV4 source config/index identity mismatch")
    config = json.loads(config_path.read_text())

    resident_path = resident_root / "resident-manifest.json"
    resident_sha = _sha256(resident_path)
    if resident_sha != expected_resident_manifest_sha256:
        raise ValueError("DSV4 resident manifest SHA-256 mismatch")
    resident = json.loads(resident_path.read_text())
    required_resident = {
        "record_type": "dsv4_resident_package_manifest_v1",
        "production_ready": True,
        "model_id": MODEL_ID,
        "source_revision": REVISION,
        "config_sha256": CONFIG_SHA256,
        "index_sha256": INDEX_SHA256,
        "backbone_tensor_count": 1564,
        "mtp_tensor_count": 97,
        "logical_parameter_tensor_count": 1271,
        "duplicate_resident_tensor_count": 0,
        "missing_resident_tensor_count": 0,
        "routed_resident_tensor_count": 0,
    }
    if any(resident.get(key) != value for key, value in required_resident.items()):
        raise ValueError("DSV4 resident manifest is not the complete pinned package")
    shards = resident.get("shards")
    if not isinstance(shards, list):
        raise TypeError("DSV4 resident shard inventory is missing")
    shard_names: list[str] = []
    for record in shards:
        if not isinstance(record, Mapping):
            raise TypeError("DSV4 resident shard inventory is invalid")
        filename = str(record.get("filename"))
        if filename in shard_names:
            raise ValueError(f"duplicate DSV4 resident shard {filename}")
        path = resident_root / filename
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != record.get("file_bytes")
        ):
            raise ValueError(f"DSV4 resident shard extent mismatch: {filename}")
        shard_names.append(filename)
    package_index = resident_root / "model.safetensors.index.json"
    if package_index.is_symlink() or _sha256(package_index) != resident.get(
        "package_index_sha256"
    ):
        raise ValueError("DSV4 resident package index mismatch")
    expected_tree = {
        "resident-manifest.json",
        "model.safetensors.index.json",
        *shard_names,
    }
    actual_tree = {path.name for path in resident_root.iterdir()}
    if actual_tree != expected_tree:
        raise ValueError("DSV4 resident artifact tree is not exact")

    vq_path = vq_root / "manifest.json"
    vq_sha = _sha256(vq_path)
    if vq_sha != expected_vq_manifest_sha256:
        raise ValueError("DSV4 VQ manifest SHA-256 mismatch")
    vq = json.loads(vq_path.read_text())
    inventory = validate_dsv4_vq_manifest_structure(vq)
    artifact_paths: dict[str, Path] = {}
    for filename, record in inventory.file_records.items():
        path = vq_root / filename
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != record["bytes"]
        ):
            raise ValueError(f"DSV4 VQ file extent mismatch: {filename}")
        artifact_paths[filename] = path

    if authenticate_payloads:
        if payload_receipt is not None or expected_payload_receipt_sha256 is not None:
            raise ValueError("DSV4 payload authentication modes are mutually exclusive")
        authenticated_receipt = authenticate_dsv4_payloads(
            resident_dir=resident_root,
            resident_manifest=resident,
            vq_dir=vq_root,
            vq_inventory=inventory,
        )
        payload_receipt_sha256 = dsv4_payload_receipt_sha256(authenticated_receipt)
    else:
        if payload_receipt is None or expected_payload_receipt_sha256 is None:
            raise ValueError("DSV4 content-hash payload receipt is required")
        payload_receipt_sha256 = validate_dsv4_payload_receipt(
            resident_dir=resident_root,
            resident_manifest=resident,
            vq_dir=vq_root,
            vq_inventory=inventory,
            receipt=payload_receipt,
            expected_receipt_sha256=expected_payload_receipt_sha256,
        )
        authenticated_receipt = payload_receipt

    identity = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "config_sha256": CONFIG_SHA256,
        "index_sha256": INDEX_SHA256,
        "resident_manifest_sha256": resident_sha,
        "resident_package_set_sha256": resident.get("package_set_sha256"),
        "vq_manifest_sha256": vq_sha,
        "vq_inventory_sha256": inventory.inventory_sha256,
        "resident_parameter_tensors": resident.get("logical_parameter_tensor_count"),
        "vq_projection_files": len(inventory.file_records),
        "payload_content_receipt_sha256": payload_receipt_sha256,
    }
    identity_sha256 = validate_dsv4_artifact_identity(identity)
    return Dsv4ValidatedCompositeInputs(
        source_dir=source_root,
        resident_dir=resident_root,
        vq_dir=vq_root,
        config=config,
        resident_manifest=resident,
        vq_manifest=vq,
        vq_inventory=inventory,
        artifact_paths=artifact_paths,
        payload_receipt=authenticated_receipt,
        payload_receipt_sha256=payload_receipt_sha256,
        identity=identity,
        identity_sha256=identity_sha256,
    )


def load_authenticated_dsv4_composite(
    inputs: Dsv4ValidatedCompositeInputs,
) -> Dsv4LoadedComposite:
    """Construct and bind exactly the authenticated resident+routed model."""

    args = deepseek_v4_flash_args_from_config(inputs.config)
    model = DeepseekV4FlashVQModel(args, with_mtp=True)
    non_vq = bind_deepseek_v4_flash_non_vq_weights(
        model, inputs.resident_dir, include_mtp=True, strict=True
    )
    backbone = bind_deepseek_v4_flash_vq_experts(
        model,
        inputs.vq_dir,
        strict=True,
        artifact_paths=inputs.artifact_paths,
    )
    mtp = bind_deepseek_v4_flash_mtp_vq_experts(
        model,
        inputs.vq_dir,
        strict=True,
        artifact_paths=inputs.artifact_paths,
    )
    dense = dense_deepseek_v4_flash_routed_parameter_names(model, include_mtp=True)
    unbound_backbone = has_unbound_deepseek_v4_flash_vq_experts(model)
    unbound_mtp = has_unbound_deepseek_v4_flash_mtp_experts(model)
    if non_vq.missing_model_parameters or dense or unbound_backbone or unbound_mtp:
        raise RuntimeError(
            "DSV4 composite bind left missing, dense, or unbound parameters"
        )
    if (
        non_vq.bound_count != 1271
        or len(set(non_vq.bound_backbone_parameters + non_vq.bound_mtp_parameters))
        != 1271
    ):
        raise RuntimeError(
            "DSV4 resident bind did not claim every parameter exactly once"
        )

    report = Dsv4CompositeLoadReport(
        artifact_identity_sha256=inputs.identity_sha256,
        non_vq_bind_report=non_vq,
        bound_backbone_layers=backbone,
        bound_mtp_stages=mtp,
        dense_routed_parameter_names=dense,
        unbound_backbone_vq_experts=unbound_backbone,
        unbound_mtp_vq_experts=unbound_mtp,
    )
    return Dsv4LoadedComposite(model=model, inputs=inputs, report=report)


__all__ = [
    "CONFIG_SHA256",
    "INDEX_SHA256",
    "MODEL_ID",
    "RESIDENT_MANIFEST_SHA256",
    "RESIDENT_PACKAGE_SET_SHA256",
    "REVISION",
    "VQ_INVENTORY_SHA256",
    "VQ_MANIFEST_SHA256",
    "Dsv4CompositeLoadReport",
    "Dsv4LoadedComposite",
    "Dsv4ValidatedCompositeInputs",
    "Dsv4VqManifestInventory",
    "authenticate_dsv4_payloads",
    "dsv4_payload_receipt_sha256",
    "load_authenticated_dsv4_composite",
    "validate_dsv4_artifact_identity",
    "validate_dsv4_composite_inputs",
    "validate_dsv4_payload_receipt",
    "validate_dsv4_vq_manifest_structure",
]
