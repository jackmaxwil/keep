from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_s3_modules_import_no_boto3_botocore_mlx_numpy_or_mlx_vq_quality() -> None:
    script = r"""
import builtins
import sys

sys.path.insert(0, SOURCE)
forbidden = ("boto3", "botocore", "mlx", "numpy", "mlx_vq")
real_import = builtins.__import__

def guarded(name, globals=None, locals=None, fromlist=(), level=0):
    if name == forbidden or any(
        name == prefix or name.startswith(prefix + ".")
        for prefix in forbidden
    ):
        raise AssertionError("forbidden import: " + name)
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded
import glm52_enforcement.s3_records
import glm52_enforcement.s3_keys
import glm52_enforcement.h1f_adapter
import glm52_enforcement.s3_adapter
loaded = set(sys.modules)
for prefix in forbidden:
    assert prefix not in loaded
    assert not any(name.startswith(prefix + ".") for name in loaded)
"""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            script.replace("SOURCE", repr(str(ROOT / "src"))),
        ],
        cwd=ROOT,
        env={"PATH": os.environ.get("PATH", "")},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
