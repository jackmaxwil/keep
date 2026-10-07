import json
import os
from dataclasses import dataclass
from pathlib import Path


def _checkpoint_path(checkpoint_dir, group_key: str) -> Path:
    return Path(checkpoint_dir) / f"{group_key}.experts.json"


def completed_experts(checkpoint_dir, group_key: str) -> frozenset[int]:
    path = _checkpoint_path(checkpoint_dir, group_key)
    if not path.exists():
        return frozenset()
    data = json.loads(path.read_text())
    return frozenset(int(e) for e in data.get("completed_experts", []))


def record_expert_done(checkpoint_dir, group_key: str, expert: int) -> None:
    path = _checkpoint_path(checkpoint_dir, group_key)
    current = set(completed_experts(checkpoint_dir, group_key))
    current.add(int(expert))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"group_key": group_key, "completed_experts": sorted(current)}))
    os.replace(tmp, path)  # atomic within the same filesystem


@dataclass(frozen=True)
class ProgressReport:
    done: int
    total: int
    seconds_elapsed: float

    @property
    def fraction(self) -> float:
        return self.done / self.total if self.total else 0.0

    @property
    def eta_seconds(self) -> float | None:
        if self.done <= 0:
            return None
        rate = self.done / self.seconds_elapsed if self.seconds_elapsed else 0.0
        if rate <= 0:
            return None
        return (self.total - self.done) / rate
