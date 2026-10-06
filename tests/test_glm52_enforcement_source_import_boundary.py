from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
MODULES = (
    ROOT / "src/glm52_enforcement/source_authorities.py",
    ROOT / "src/glm52_enforcement/source_publishers.py",
)
BLOCKED = {"mlx", "numpy", "mlx_vq", "boto3", "botocore"}


def test_source_modules_import_no_mlx_numpy_mlx_vq_boto3_or_botocore() -> None:
    imported_roots = set()
    for path in MODULES:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(
                    alias.name.split(".", 1)[0] for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                assert node.module is not None
                imported_roots.add(node.module.split(".", 1)[0])
    assert imported_roots.isdisjoint(BLOCKED)

    script = """
import importlib.abc
import sys

blocked = {"mlx", "numpy", "mlx_vq", "boto3", "botocore"}
seen = []

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".", 1)[0]
        if root in blocked:
            seen.append(fullname)
            raise AssertionError("blocked import executed: " + fullname)
        return None

sys.meta_path.insert(0, Blocker())
import glm52_enforcement.source_authorities
import glm52_enforcement.source_publishers
executed = {name.split(".", 1)[0] for name in sys.modules}
assert blocked.isdisjoint(executed)
assert seen == []
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, (
        completed.stdout,
        completed.stderr,
    )
