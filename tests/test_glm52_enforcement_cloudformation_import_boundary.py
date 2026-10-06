from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
MODULES = (
    ROOT / "src/glm52_enforcement/cloudformation_stacks.py",
    ROOT / "src/glm52_enforcement/fence_executor.py",
)
BLOCKED = {"mlx", "numpy", "mlx_vq", "boto3", "botocore"}


def test_cloudformation_modules_import_no_mlx_numpy_mlx_vq_boto3_or_botocore() -> None:
    for path in MODULES:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".", 1)[0])
        assert imported.isdisjoint(BLOCKED), (path, imported & BLOCKED)

    script = """
import builtins
import sys

blocked = {"mlx", "numpy", "mlx_vq", "boto3", "botocore"}
original = builtins.__import__
def guarded(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split(".", 1)[0] in blocked:
        raise AssertionError("blocked import attempted: " + name)
    return original(name, globals, locals, fromlist, level)
builtins.__import__ = guarded
import glm52_enforcement.cloudformation_stacks
import glm52_enforcement.fence_executor
assert blocked.isdisjoint(sys.modules)
print("task6-cloudformation-import-boundary-ok")
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "task6-cloudformation-import-boundary-ok"
