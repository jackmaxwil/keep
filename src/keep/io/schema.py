from __future__ import annotations

from dataclasses import dataclass, field

from keep.vq.e8 import E8_1BIT_PACKED_SHA256, E8P_PACKED_ABS_SHA256


SUPPORTED_CODE_BITS = {8, 16}
DEFAULT_GROUP_SIZE = 512


def codebook_metadata_for_bits(code_bits: int) -> tuple[str, str]:
    if code_bits == 8:
        return "quip_e8", E8_1BIT_PACKED_SHA256
    if code_bits == 16:
        return "quip_e8p", E8P_PACKED_ABS_SHA256
    raise ValueError("code_bits must be 8 or 16")


@dataclass(frozen=True)
class VQTensorSpec:
    name: str
    in_dim: int
    out_dim: int
    code_bits: int = 8
    group_size: int = DEFAULT_GROUP_SIZE
    num_experts: int | None = None

    def validate(self) -> None:
        if self.code_bits not in SUPPORTED_CODE_BITS:
            raise ValueError(f"{self.name}: code_bits must be one of {sorted(SUPPORTED_CODE_BITS)}")
        if self.in_dim <= 0 or self.out_dim <= 0:
            raise ValueError(f"{self.name}: dimensions must be positive")
        if self.in_dim % 8 != 0:
            raise ValueError(f"{self.name}: in_dim must be divisible by 8")
        if self.group_size <= 0:
            raise ValueError(f"{self.name}: group_size must be positive")
        if self.in_dim % self.group_size != 0:
            raise ValueError(f"{self.name}: in_dim must be divisible by group_size")
        if self.num_experts is not None and self.num_experts <= 0:
            raise ValueError(f"{self.name}: num_experts must be positive when provided")

    @property
    def codeword_dtype(self) -> str:
        return "uint8" if self.code_bits == 8 else "uint16"

    @property
    def codes_shape(self) -> tuple[int, ...]:
        self.validate()
        shape = (self.out_dim, self.in_dim // 8)
        if self.num_experts is not None:
            return (self.num_experts, *shape)
        return shape

    @property
    def scales_shape(self) -> tuple[int, ...]:
        self.validate()
        shape = (self.out_dim, self.in_dim // self.group_size)
        if self.num_experts is not None:
            return (self.num_experts, *shape)
        return shape


@dataclass(frozen=True)
class QuantizationConfig:
    quant_method: str = "mlx_vq_e8"
    version: int = 1
    default_code_bits: int = 8
    default_group_size: int = DEFAULT_GROUP_SIZE
    codebook_name: str = "quip_e8"
    codebook_entries: int = 256
    codebook_sha256: str = E8_1BIT_PACKED_SHA256
    policy: dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        if self.quant_method != "mlx_vq_e8":
            raise ValueError(f"unsupported quant_method {self.quant_method!r}")
        if self.version != 1:
            raise ValueError(f"unsupported quantization config version {self.version}")
        if self.default_code_bits not in SUPPORTED_CODE_BITS:
            raise ValueError("default_code_bits must be 8 or 16")
        if self.default_group_size <= 0:
            raise ValueError("default_group_size must be positive")
        if self.codebook_entries != 256:
            raise ValueError("E8 codebook must have 256 entries")
        _, expected_hash = codebook_metadata_for_bits(self.default_code_bits)
        if self.codebook_sha256 != expected_hash:
            raise ValueError("E8 codebook sha256 does not match default_code_bits")
        if len(self.codebook_sha256) != 64:
            raise ValueError("E8 codebook sha256 must be a 64-character hex digest")

    def to_json_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "quant_method": self.quant_method,
            "version": self.version,
            "default_code_bits": self.default_code_bits,
            "default_group_size": self.default_group_size,
            "codebook": {
                "name": self.codebook_name,
                "dtype": "uint32",
                "entries": self.codebook_entries,
                "sha256": self.codebook_sha256,
            },
            "policy": dict(self.policy),
        }
