"""MLX/Metal VQ kernel entry points."""

__all__ = ["gather_vqmm_kernel", "vq_qmm", "vq_qmm_reference_np", "vq_qmv", "vq_qmv_reference_np"]

_EXPORT_MODULES = {
    "gather_vqmm_kernel": "mlx_vq.kernels.gather_vqmm",
    "vq_qmm": "mlx_vq.kernels.vq_qmm",
    "vq_qmm_reference_np": "mlx_vq.kernels.vq_qmm",
    "vq_qmv": "mlx_vq.kernels.vq_qmv",
    "vq_qmv_reference_np": "mlx_vq.kernels.vq_qmv",
}


def __getattr__(name: str):
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name = _EXPORT_MODULES[name]
    module = __import__(module_name, fromlist=[name])
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
