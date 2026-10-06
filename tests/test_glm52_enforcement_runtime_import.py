from __future__ import annotations

from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_target_import_blocks_mlx_numpy_boto3_botocore_and_mlx_vq() -> None:
    code = """
import sys
import glm52_enforcement
assert glm52_enforcement.__all__ == ["__version__"]
for name in (
    "glm52_enforcement.approvals",
    "glm52_enforcement.canonical",
    "glm52_enforcement.records",
    "glm52_enforcement.transitions",
    "mlx", "numpy", "boto3", "botocore", "mlx_vq",
):
    assert name not in sys.modules, name
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=REPO_ROOT,
        env={"PYTHONPATH": str(REPO_ROOT / "src")},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_python39_compiles_the_complete_standalone_package() -> None:
    python = Path("/usr/bin/python3")
    if not python.exists():
        return
    version = subprocess.run(
        [str(python), "-c", "import sys; print(sys.version_info[:2])"],
        text=True,
        capture_output=True,
        check=True,
    )
    assert version.stdout.strip() == "(3, 9)"
    result = subprocess.run(
        [str(python), "-m", "compileall", "-q", str(REPO_ROOT / "src/glm52_enforcement")],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
