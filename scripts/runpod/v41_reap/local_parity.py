"""Run an exported REAP checkpoint on CPU and score it against the pod's teacher, per position.

The Runpod eval scored each candidate by masking pruned experts out of the full model's
router (``collect.py --survivors``). This script runs the exported checkpoint itself, which
has those experts physically removed and the router sliced, through the same layer-streamed
reference code with the CPU kernel stand-ins in ``torch_kernel.py``. It reads the teacher's
top-K logits from the pod's eval npz files, so the full model is never needed here, then
compares KL, NLL and top-1 agreement with the pod's masked-router numbers for the same
positions.

    python local_parity.py --hf-path ~/Models/v41-reap75 \\
        --pack ~/Models/v41-reap-run/pack/v41-pack.jsonl \\
        --stats ~/Models/v41-reap-run/stats --keep 96 \\
        --sessions eval-00054 eval-00100 --out ~/Models/v41-reap-run/parity-96.json
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import threading
import time
from pathlib import Path

import numpy as np

KIT = Path(__file__).resolve().parent
sys.path.insert(0, str(KIT))


def rss_guard(limit_gb: float) -> None:
    """Exit hard if this process's peak resident size passes ``limit_gb`` (Jack's memory budget)."""

    def watch() -> None:
        while True:
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30  # bytes on macOS
            if peak > limit_gb:
                print(f"peak RSS {peak:.1f} GB passed the {limit_gb} GB budget; exiting", flush=True)
                os._exit(3)
            time.sleep(2)

    threading.Thread(target=watch, daemon=True).start()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hf-path", required=True, type=Path, help="the exported REAP checkpoint")
    p.add_argument("--pack", required=True, type=Path)
    p.add_argument("--stats", required=True, type=Path, help="local copy of the REAP-stats dataset (eval/ and student-KEEP/)")
    p.add_argument("--keep", required=True, help="the pod candidate to compare with, e.g. 96")
    p.add_argument("--sessions", nargs="+", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--max-rss-gb", type=float, default=24.0)
    p.add_argument("--threads", type=int, default=os.cpu_count())
    args = p.parse_args(argv)
    rss_guard(args.max_rss_gb)

    import torch
    from transformers import AutoTokenizer

    import collect
    import stream
    import torch_kernel

    torch_kernel.exact_quant = True
    torch.set_num_threads(args.threads)
    ref_dir = args.hf_path / "inference"
    ref = stream.import_reference(ref_dir, kernel=torch_kernel)
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(33377335)
    model_args = ref.ModelArgs(**json.loads((ref_dir / "config.json").read_text()))
    model_args.max_batch_size = 1

    sessions = collect.load_sessions(args.pack)
    records = {name: sessions[name][0] for name in args.sessions}
    model_args.max_seq_len = max(len(r["token_ids"]) for r in records.values())

    ckpt = stream.HFCheckpoint(args.hf_path)
    layout = ref.EngramLayout.from_args(model_args)
    hash_state = ref.NgramHashState(model_args, layout, AutoTokenizer.from_pretrained(args.hf_path))
    engram_rows = {l: stream.EngramRows.from_checkpoint(ckpt, l, ref.fp8_block_size) for l in model_args.engram_layer_ids}
    streamer = stream.LayerStreamer(ref, model_args, ckpt.load, engram_rows, hash_state, "cpu")
    state = collect.install_hooks(streamer, top_k=2048, rows_per_slice=256)

    items = {name: stream.Item(tokens=torch.as_tensor([r["token_ids"]], dtype=torch.long), group=("student", None))  # one MoE pass for all
             for name, r in records.items()}
    t0 = time.time()
    with torch.inference_mode():
        streamer.forward(list(items.values()))
    forward_s = time.time() - t0

    report: dict[str, dict[str, float]] = {}
    for name, record in records.items():
        teacher = np.load(args.stats / "eval" / "sessions" / f"{name}.npz")
        pod = np.load(args.stats / f"student-{args.keep}" / "eval" / "sessions" / f"{name}.npz")
        positions = collect.target_positions(record)
        assert np.array_equal(positions, teacher["positions"]) and np.array_equal(positions, pod["positions"]), name
        targets = np.asarray(record["token_ids"], dtype=np.int64)[positions + 1]
        state["positions"] = torch.as_tensor(positions)
        state["targets"] = torch.as_tensor(targets)
        state["teacher"] = {
            "ids": torch.as_tensor(teacher["topk_logit_ids"].astype(np.int64)),
            "logprob": torch.as_tensor(teacher["topk_logit_values"].astype(np.float32) - teacher["logsumexp"][:, None]),
            "targets": state["targets"],
        }
        state["captured"] = {k: [] for k in collect.CAPTURE_KEYS}
        with torch.inference_mode():
            streamer.head(streamer.final_hidden(items[name]))
        cap = {k: np.concatenate(v) for k, v in state["captured"].items() if v}
        teacher_nll = teacher["logsumexp"] - teacher["target_logit"]
        row = {"domain": record["domain"], "positions": int(positions.size)}
        for key, local, remote in (
            ("kl", cap["kl_on_teacher_topk"], pod["kl_on_teacher_topk"]),
            ("nll_delta", cap["student_nll"] - teacher_nll, pod["student_nll"] - teacher_nll),
            ("top1_agree", cap["top1_agree"].astype(np.float64), pod["top1_agree"].astype(np.float64)),
        ):
            row[f"{key}_local"] = float(np.mean(local))
            row[f"{key}_pod"] = float(np.mean(remote))
            row[f"{key}_mean_abs_diff"] = float(np.mean(np.abs(local - remote)))
        report[name] = row
        print(json.dumps({name: row}), flush=True)

    summary = {
        "hf_path": str(args.hf_path),
        "keep": args.keep,
        "forward_seconds": round(forward_s, 1),
        "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 2),
        "sessions": report,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "sessions"}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
