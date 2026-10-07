"""Write the two-rank JACCL hostfile used by GLM-4.5-Air clean-room runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def write_two_rank_hostfile(
    *,
    output_dir: Path,
    local_ip: str,
    local_rdma_device: str,
    peer_ssh: str,
    peer_rdma_device: str,
    local_ssh: str = "127.0.0.1",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    hostfile = output_dir / "hostfile.json"
    rows = [
        {
            "ssh": local_ssh,
            "ips": [local_ip],
            "rdma": [None, local_rdma_device],
        },
        {
            "ssh": peer_ssh,
            "ips": [],
            "rdma": [peer_rdma_device, None],
        },
    ]
    hostfile.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return hostfile


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write a two-rank MLX/JACCL hostfile for clean-room cache runs."
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--local-ssh", default="127.0.0.1")
    parser.add_argument("--local-ip", required=True)
    parser.add_argument("--local-rdma-device", required=True)
    parser.add_argument("--peer-ssh", required=True)
    parser.add_argument("--peer-rdma-device", required=True)
    args = parser.parse_args()

    hostfile = write_two_rank_hostfile(
        output_dir=args.output_dir,
        local_ssh=args.local_ssh,
        local_ip=args.local_ip,
        local_rdma_device=args.local_rdma_device,
        peer_ssh=args.peer_ssh,
        peer_rdma_device=args.peer_rdma_device,
    )
    print(hostfile)


if __name__ == "__main__":
    main()
