"""Teacher pass for DeepSeek-V4.1-Flash on a GPU node: REAP, imatrix, routes, logits.

Runs DeepSeek's own reference inference code (``inference/model.py`` at the
pinned revision) under ``torchrun``, with one change: ``MoE.forward`` is
replaced by a copy that also records, for every routed slot,

- REAP statistics (``keep.quality.reap.ReapStats`` arrays, ``reap__*``);
- imatrix sums for the two expert input spaces (``hidden`` = the post
  ``ffn_norm`` row that w1/w3 read, ``down`` = the GLU activation w2 reads),
  under the keys ``keep.quality.dsv4_teacher_runner`` finalizes;
- the first ``ROUTE_RECORD_TOKENS`` routes of each session (``routes__*``),
  for ``benchmarks/replay_dsv4_expert_residency.py``.

Eval sessions also capture top-K teacher logits at their supervised positions,
in the teacher runner's ``logits`` schema. Output, per session, is one npz under
``<out>/<split>/sessions/``, so KEEP's existing ``finalize-reap`` and imatrix
finalizers read it unchanged.

The replacement forward computes exactly what the reference does (same ops,
same order); ``tests/test_runpod_v41_reap.py`` checks it, bit for bit,
against a CPU copy of the reference Gate / Expert / MoE. One approximation: the reference folds the gate weight
in *before* w2 (``w2(g * h)``), so the expert output norm ``||f||`` is taken as
``||w2(g * h)|| / g``. With FP8 activation quantization in w2 that differs
from ``||w2(h)||`` by rounding only.

Usage (on the pod, from the reference ``inference`` directory)::

    torchrun --nproc-per-node 8 /workspace/keep/scripts/runpod/v41_reap/collect.py \\
        --ckpt-path /workspace/v41-tp8 --config config.json \\
        --pack /workspace/pack/v41-pack.jsonl --out /workspace/out \\
        --deadline-utc 2026-11-07T06:00:00Z

Then, after finalize-reap and reap-select on the teacher pass, the same command
with ``--survivors 192=/workspace/out/survivors-192.json`` (repeatable) scores
each candidate expert count against the teacher in ``<out>/student-KEEP/``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

MODEL_ID = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
PINNED_REVISION = "1bae0e1088993b9b82de08f59de5be5a34e42170"
#: sha256 of inference/model.py at PINNED_REVISION. The patched forward below
#: mirrors that file's MoE.forward and Expert.forward; refuse anything else.
REFERENCE_MODEL_PY_SHA256 = "e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d"
ROUTE_RECORD_TOKENS = 2048  # matches keep.quality.expert_residency
REAP_PREFIX = "reap__"
ROUTE_PREFIX = "routes__"
SPACES = ("hidden", "down")
CAPTURE_KEYS = (
    "topk_logit_ids", "topk_logit_values", "logsumexp", "tail_mass", "target_logit",  # teacher pass
    "kl_on_teacher_topk", "student_nll", "top1_agree",  # masked-router student pass
)


# ---------------------------------------------------------------------------
# Accumulation (torch, float64 on device; one instance per process)
# ---------------------------------------------------------------------------


class SessionStats:
    """Per-session sums over this rank's local experts; all-reduced at the end."""

    def __init__(self, *, torch: Any, n_layers: int, n_experts: int, dims: dict[str, int], device: Any):
        self.torch = torch
        self.n_layers, self.n_experts = n_layers, n_experts
        f64 = dict(dtype=torch.float64, device=device)
        z = lambda *shape: torch.zeros(shape, **f64)
        self.reap = {
            "route_count": z(n_layers, n_experts),
            "gate_sum": z(n_layers, n_experts),
            "gate_norm_sum": z(n_layers, n_experts),
            "norm_sum": z(n_layers, n_experts),
            "norm_max": z(n_layers, n_experts),
            "abs_max": z(n_layers, n_experts),
        }
        self.routed_slots = z(n_layers)
        self.imatrix = {
            space: {
                "importance_sum": z(n_layers, n_experts, dim),
                "affinity_weighted_importance": z(n_layers, n_experts, dim),
                "route_count": z(n_layers, n_experts),
                "affinity_score_sum": z(n_layers, n_experts),
                "total_route_count": z(n_layers),
            }
            for space, dim in dims.items()
        }
        self.route_ids: list[list[np.ndarray]] = [[] for _ in range(n_layers)]
        self.route_gates: list[list[np.ndarray]] = [[] for _ in range(n_layers)]
        self.route_tokens = [0] * n_layers

    def add_routes(self, layer: int, indices: Any, weights: Any, *, record: bool) -> None:
        """Once per layer per forward, on every rank (the gate is replicated)."""

        rows, top_k = indices.shape
        if record:  # rank 0 only: identical on every rank
            self.routed_slots[layer] += rows * top_k
            for space in SPACES:
                self.imatrix[space]["total_route_count"][layer] += rows * top_k
            room = ROUTE_RECORD_TOKENS - self.route_tokens[layer]
            if room > 0:
                self.route_ids[layer].append(indices[:room].to("cpu").numpy().astype(np.int16))
                self.route_gates[layer].append(weights[:room].float().to("cpu").numpy().astype(np.float16))
                self.route_tokens[layer] += min(room, rows)

    def add_expert(self, layer: int, expert: int, x: Any, gate: Any, out: Any, h: Any) -> None:
        """One local expert's routed rows: input ``x``, gate ``gate`` [r], output, GLU ``h``."""

        torch = self.torch
        g = gate.double()
        out64 = out.double()
        weighted_norm = out64.norm(dim=-1)  # ||g f|| (g > 0)
        f_norm = weighted_norm / g
        r = self.reap
        r["route_count"][layer, expert] += g.numel()
        r["gate_sum"][layer, expert] += g.sum()
        r["gate_norm_sum"][layer, expert] += weighted_norm.sum()
        r["norm_sum"][layer, expert] += f_norm.sum()
        r["norm_max"][layer, expert] = torch.maximum(r["norm_max"][layer, expert], f_norm.max())
        abs_max = (out64.abs().amax(dim=-1) / g).max()
        r["abs_max"][layer, expert] = torch.maximum(r["abs_max"][layer, expert], abs_max)
        for space, rows in (("hidden", x), ("down", h)):
            sq = rows.double().square()
            m = self.imatrix[space]
            m["importance_sum"][layer, expert] += sq.sum(dim=0)
            m["affinity_weighted_importance"][layer, expert] += (sq * g[:, None]).sum(dim=0)
            m["route_count"][layer, expert] += g.numel()
            m["affinity_score_sum"][layer, expert] += g.sum()

    def reduce(self, dist: Any) -> None:
        if dist is None or not dist.is_initialized():
            return
        for name, value in self.reap.items():
            op = dist.ReduceOp.MAX if name in ("norm_max", "abs_max") else dist.ReduceOp.SUM
            dist.all_reduce(value, op=op)
        dist.all_reduce(self.routed_slots)
        for space in SPACES:
            for value in self.imatrix[space].values():
                dist.all_reduce(value)

    def arrays(self, layers: list[int]) -> dict[str, np.ndarray]:
        np_ = lambda t: t.to("cpu").numpy()
        out: dict[str, np.ndarray] = {
            "layers": np.asarray(layers, dtype=np.int32),
            "num_experts": np.asarray(self.n_experts, dtype=np.int32),
            f"{REAP_PREFIX}layers": np.asarray(layers, dtype=np.int64),
            f"{REAP_PREFIX}num_experts": np.asarray(self.n_experts, dtype=np.int64),
            f"{REAP_PREFIX}routed_slots": np.rint(np_(self.routed_slots)).astype(np.int64),
        }
        for name, value in self.reap.items():
            arr = np_(value)
            out[f"{REAP_PREFIX}{name}"] = np.rint(arr).astype(np.int64) if name == "route_count" else arr
        for space in SPACES:
            m = self.imatrix[space]
            # float32 on disk, as KEEP's validator requires (sums stay float64 until here)
            out[f"importance_sum__{space}"] = np_(m["importance_sum"]).astype(np.float32)
            out[f"affinity_weighted_importance__{space}"] = np_(m["affinity_weighted_importance"]).astype(np.float32)
            out[f"route_count__{space}"] = np.rint(np_(m["route_count"])).astype(np.int64)
            out[f"affinity_score_sum__{space}"] = np_(m["affinity_score_sum"])
            out[f"total_route_count__{space}"] = np.rint(np_(m["total_route_count"])).astype(np.int64)
        if all(self.route_ids[layer] for layer in range(self.n_layers)):
            n = min(self.route_tokens)
            out[f"{ROUTE_PREFIX}layers"] = np.asarray(layers, dtype=np.int64)
            out[f"{ROUTE_PREFIX}expert_ids"] = np.stack(
                [np.concatenate(self.route_ids[layer])[:n] for layer in range(self.n_layers)], axis=1
            )
            out[f"{ROUTE_PREFIX}gate_weights"] = np.stack(
                [np.concatenate(self.route_gates[layer])[:n] for layer in range(self.n_layers)], axis=1
            )
        return out


# ---------------------------------------------------------------------------
# Patched reference modules
# ---------------------------------------------------------------------------


def expert_forward_with_activation(F: Any, torch: Any, expert: Any, x: Any, weights: Any) -> tuple[Any, Any]:
    """``Expert.forward(x, weights)`` from the reference, also returning the GLU activation."""

    dtype = x.dtype
    gate = expert.w1(x).float()
    up = expert.w3(x).float()
    if expert.swiglu_limit > 0:
        up = torch.clamp(up, min=-expert.swiglu_limit, max=expert.swiglu_limit)
        gate = torch.clamp(gate, max=expert.swiglu_limit)
    h = F.silu(gate) * up
    return expert.w2((weights * h).to(dtype)), h


def make_moe_forward(ref: Any, torch: Any, F: Any, dist: Any, state: dict[str, Any]):
    """A drop-in ``MoE.forward`` that records into ``state['stats']`` when set."""

    def forward(self, x, image_mask=None):
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x, None if image_mask is None else image_mask.flatten())
        y = torch.zeros_like(x, dtype=torch.float32)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        stats = state.get("stats")
        layer = self.layer_id if stats is not None and self.layer_id < stats.n_layers else None
        if layer is not None:
            stats.add_routes(layer, indices, weights, record=state["rank"] == 0)
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            w = weights[idx, top, None]
            if layer is None:
                y[idx] += expert(x[idx], w)
                continue
            out, h = expert_forward_with_activation(F, torch, expert, x[idx], w)
            stats.add_expert(layer, i, x[idx], w[:, 0], out, h)
            y[idx] += out
        if ref.world_size > 1:
            dist.all_reduce(y)
        y += self.shared_experts(x)
        return y.type_as(x).view(shape)

    return forward


def make_head_forward(head: Any, torch: Any, F: Any, dist: Any, state: dict[str, Any], *, top_k: int, rows_per_slice: int):
    """``ParallelHead.forward`` that also captures top-K logits at ``state['positions']``."""

    original = head.forward

    def forward(x, full_logits=False):
        positions = state.get("positions")
        if positions is not None:
            picked = x[0, positions]  # [P, dim], already normalized
            for start in range(0, picked.shape[0], rows_per_slice):
                part = F.linear(picked[start : start + rows_per_slice].float(), head.weight)
                if dist is not None and dist.is_initialized() and dist.get_world_size() > 1:
                    gathered = [torch.empty_like(part) for _ in range(dist.get_world_size())]
                    dist.all_gather(gathered, part)
                    part = torch.cat(gathered, dim=-1)
                lse = torch.logsumexp(part, dim=-1)
                teacher = state.get("teacher")
                if teacher is not None:  # masked-router student pass: score against the teacher, keep scalars
                    rows = slice(start, start + part.shape[0])
                    t_logp = teacher["logprob"][rows]
                    s_logp = part.gather(1, teacher["ids"][rows]) - lse[:, None]
                    kl = (t_logp.exp() * (t_logp - s_logp)).sum(dim=-1)
                    nll = lse - part.gather(1, teacher["targets"][rows, None])[:, 0]
                    agree = part.argmax(dim=-1) == teacher["ids"][rows, 0]
                    if state["rank"] == 0:
                        cap = state["captured"]
                        cap["kl_on_teacher_topk"].append(kl.cpu().numpy().astype(np.float32))
                        cap["student_nll"].append(nll.cpu().numpy().astype(np.float32))
                        cap["top1_agree"].append(agree.cpu().numpy())
                    continue
                values, ids = part.topk(top_k, dim=-1)
                tail = (1.0 - torch.exp(values - lse[:, None]).sum(dim=-1)).clamp_min(0.0)
                target = part.gather(1, state["targets"][start : start + part.shape[0], None])[:, 0]
                if state["rank"] == 0:
                    cap = state["captured"]
                    cap["target_logit"].append(target.cpu().numpy().astype(np.float32))
                    cap["topk_logit_ids"].append(ids.to(torch.int32).cpu().numpy())
                    cap["topk_logit_values"].append(values.cpu().numpy().astype(np.float16))
                    cap["logsumexp"].append(lse.cpu().numpy().astype(np.float32))
                    cap["tail_mass"].append(tail.cpu().numpy().astype(np.float16))
        return original(x, full_logits)

    return forward


# ---------------------------------------------------------------------------
# Pack and output
# ---------------------------------------------------------------------------


def load_sessions(pack: Path) -> OrderedDict[str, list[dict[str, Any]]]:
    """Pack lines grouped by ``session`` in file order.

    Each line: ``{"id", "session", "split": "calibration"|"eval", "domain",
    "token_ids": [...], "target_spans": [[start, end], ...]}``. Target spans
    index ``token_ids``; logits are captured at ``span_index - 1``.
    """

    sessions: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    with pack.open() as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                sessions.setdefault(record["session"], []).append(record)
    for name, records in sessions.items():
        splits = {r["split"] for r in records}
        if len(splits) != 1 or not splits <= {"calibration", "eval"}:
            raise ValueError(f"session {name} mixes or misnames splits: {sorted(splits)}")
        if splits == {"eval"} and len(records) != 1:
            raise ValueError(f"eval session {name} must hold exactly one sample (one logits stream)")
    return sessions


def target_positions(record: dict[str, Any]) -> np.ndarray:
    n = len(record["token_ids"])
    targets = sorted({t for start, end in record["target_spans"] for t in range(max(int(start), 1), min(int(end), n))})
    return np.asarray(targets, dtype=np.int64) - 1


def sha256_ids(ids: list[int]) -> str:
    return hashlib.sha256(np.asarray(ids, dtype=np.int32).tobytes()).hexdigest()


def identity(*, session: str, split: str, records: list[dict[str, Any]], config_sha: str, chunk: int) -> dict[str, np.ndarray]:
    ids_hash = hashlib.sha256("".join(sha256_ids(r["token_ids"]) for r in records).encode()).hexdigest()
    mode = "calibration" if split == "calibration" else "logits"
    return {
        "record_type": np.asarray(f"dsv4_teacher_{mode}_v1"),
        "schema_version": np.asarray(1, dtype=np.int32),
        "prompt_id": np.asarray(session),
        "campaign_split": np.asarray(split),
        "token_ids_sha256": np.asarray(ids_hash),
        "token_count": np.asarray(sum(len(r["token_ids"]) for r in records), dtype=np.int64),
        "prefill_chunk_tokens": np.asarray(chunk, dtype=np.int32),
        "generation_config_sha256": np.asarray(config_sha),
        "producer": np.asarray("keep/scripts/runpod/v41_reap/collect.py (DeepSeek reference torch)"),
        "model_id": np.asarray(MODEL_ID),
        "model_revision": np.asarray(PINNED_REVISION),
        "domains": np.asarray(sorted({r["domain"] for r in records})),
        "sample_ids": np.asarray([r["id"] for r in records]),
    }


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt-path", required=True, help="convert.py output: model{rank}-mp{N}.safetensors + tokenizer")
    parser.add_argument("--config", default="config.json", help="reference inference/config.json")
    parser.add_argument("--pack", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-seq-len", type=int, default=16384)
    parser.add_argument("--top-k", type=int, default=2048, help="teacher logits kept per eval position")
    parser.add_argument("--rows-per-slice", type=int, default=256)
    parser.add_argument("--deadline-utc", help="stop cleanly after the session running at this time")
    parser.add_argument(
        "--survivors", action="append", default=[], metavar="KEEP=PATH",
        help="student pass (repeatable): mask the router to a reap-select table, score eval sessions "
        "against the teacher run in --out, write <out>/student-KEEP/",
    )
    parser.add_argument("--skip-hash-check", action="store_true", help="tests only")
    args = parser.parse_args(argv)

    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from safetensors.torch import load_model
    from transformers import AutoTokenizer

    here = Path.cwd()
    model_py = here / "model.py"
    digest = hashlib.sha256(model_py.read_bytes()).hexdigest()
    if not args.skip_hash_check and digest != REFERENCE_MODEL_PY_SHA256:
        raise SystemExit(f"{model_py} sha256 {digest} is not the pinned reference ({REFERENCE_MODEL_PY_SHA256})")
    sys.path.insert(0, str(here))
    import model as ref

    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(33377335)

    with open(args.config) as f:
        config = json.load(f)
    model_args = ref.ModelArgs(**config)
    model_args.max_batch_size = 1
    model_args.max_seq_len = args.max_seq_len
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_path)
    with torch.device("cuda"):
        model = ref.Transformer(model_args, tokenizer)
    load_model(model, os.path.join(args.ckpt_path, f"model{rank}-mp{world_size}.safetensors"))
    torch.set_default_device("cuda")
    model.eval()
    original_bias = {
        (layer, leaf): getattr(block.ffn.gate, leaf).data.clone()
        for layer, block in enumerate(model.layers)
        for leaf in ("bias", "bias_vl")
        if getattr(block.ffn.gate, leaf) is not None
    }

    def mask_router(table_path: Path | None) -> str | None:
        """Masking the router to -inf equals slicing it (tests/test_reap.py), so one
        loaded checkpoint scores every candidate expert count."""

        for (layer, leaf), value in original_bias.items():
            getattr(model.layers[layer].ffn.gate, leaf).data.copy_(value)
        if table_path is None:
            return None
        for layer, kept in json.loads(table_path.read_text())["survivors"].items():
            gate = model.layers[int(layer)].ffn.gate
            mask = torch.ones(gate.bias.shape[0], dtype=torch.bool, device=gate.bias.device)
            mask[torch.as_tensor(kept, device=mask.device)] = False
            for leaf in ("bias", "bias_vl"):
                if (int(layer), leaf) in original_bias:
                    getattr(gate, leaf).data[mask] = float("-inf")
        return hashlib.sha256(table_path.read_bytes()).hexdigest()

    state: dict[str, Any] = {"rank": rank, "stats": None, "positions": None}
    ref.MoE.forward = make_moe_forward(ref, torch, F, dist, state)
    model.head.forward = make_head_forward(model.head, torch, F, dist, state, top_k=args.top_k, rows_per_slice=args.rows_per_slice)

    run_config = {
        "model_id": MODEL_ID,
        "revision": PINNED_REVISION,
        "model_py_sha256": digest,
        "pack_sha256": hashlib.sha256(args.pack.read_bytes()).hexdigest(),
        "max_seq_len": args.max_seq_len,
        "top_k": args.top_k,
        "world_size": world_size,
    }
    config_sha = hashlib.sha256(json.dumps(run_config, sort_keys=True).encode()).hexdigest()
    if rank == 0:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "run-config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n")

    deadline = None
    if args.deadline_utc:
        deadline = dt.datetime.fromisoformat(args.deadline_utc.replace("Z", "+00:00")).timestamp()
    n_layers = int(model_args.n_layers)
    n_experts = int(model_args.n_routed_experts)
    dims = {"hidden": int(model_args.dim), "down": int(model_args.moe_inter_dim)}
    layers = list(range(n_layers))
    sessions = load_sessions(args.pack)
    started = time.time()
    teacher_config_sha = hashlib.sha256((args.out / "run-config.json").read_bytes()).hexdigest() if rank == 0 else ""

    def run_pass(out_dir: Path, survivors_sha: str | None) -> bool:
        """One pass over the pack. Teacher pass when ``survivors_sha`` is None, else a
        masked-router student pass over the eval sessions. False once the deadline hits."""

        student = survivors_sha is not None
        progress = out_dir / "progress.jsonl"
        for name, records in sessions.items():
            split = records[0]["split"]
            target = out_dir / split / "sessions" / f"{name}.npz"
            if target.exists() or (student and split != "eval"):
                continue
            if deadline is not None and time.time() > deadline:
                if rank == 0:
                    print(f"deadline reached before session {name}; stopping cleanly", flush=True)
                return False
            t0 = time.time()
            stats = None if student else SessionStats(torch=torch, n_layers=n_layers, n_experts=n_experts, dims=dims, device="cuda")
            state["stats"] = stats
            state["captured"] = {k: [] for k in CAPTURE_KEYS}
            if student:
                with np.load(args.out / "eval" / "sessions" / f"{name}.npz") as teacher:
                    state["teacher"] = {
                        "ids": torch.as_tensor(teacher["topk_logit_ids"].astype(np.int64), device="cuda"),
                        "logprob": torch.as_tensor(
                            teacher["topk_logit_values"].astype(np.float32) - teacher["logsumexp"][:, None], device="cuda"
                        ),
                        "targets": torch.as_tensor(teacher["target_token_ids"].astype(np.int64), device="cuda"),
                    }
            tokens_seen = 0
            logits_arrays: dict[str, np.ndarray] = {}
            for record in records:
                ids = record["token_ids"]
                if len(ids) > args.max_seq_len:
                    raise ValueError(f"{record['id']} has {len(ids)} tokens > --max-seq-len")
                positions = target_positions(record) if split == "eval" else None
                state["positions"] = None if positions is None else torch.as_tensor(positions, device="cuda")
                if positions is not None:
                    state["targets"] = torch.as_tensor(np.asarray(ids, dtype=np.int64)[positions + 1], device="cuda")
                with torch.inference_mode():
                    model.forward(torch.as_tensor([ids], dtype=torch.long, device="cuda"), 0)
                tokens_seen += len(ids)
                if positions is not None:
                    logits_arrays = {
                        "positions": positions.astype(np.int32),
                        "target_token_ids": np.asarray(ids, dtype=np.int32)[positions + 1],
                    }
                    if rank == 0:
                        logits_arrays.update({k: np.concatenate(v) for k, v in state["captured"].items() if v})
            state["positions"] = state["targets"] = state["teacher"] = state["stats"] = None
            if stats is not None:
                stats.reduce(dist)
            if rank == 0:
                arrays = identity(session=name, split=split, records=records, config_sha=config_sha, chunk=args.max_seq_len)
                if stats is not None:
                    stat_arrays = stats.arrays(layers)
                    if split == "eval":  # REAP + routes only: per-split saliency, not ~0.9 GB of imatrix per sample
                        stat_arrays = {k: v for k, v in stat_arrays.items() if k.split("__")[-1] not in SPACES}
                    arrays.update(stat_arrays)
                arrays.update(logits_arrays)
                if student:
                    arrays["record_type"] = np.asarray("dsv4_masked_student_eval_v1")
                    arrays["survivors_sha256"] = np.asarray(survivors_sha)
                    arrays["teacher_run_config_sha256"] = np.asarray(teacher_config_sha)
                save_npz(target, arrays)
                entry = {
                    "session": name,
                    "split": split,
                    "tokens": tokens_seen,
                    "seconds": round(time.time() - t0, 3),
                    "tok_per_s": round(tokens_seen / max(time.time() - t0, 1e-9), 1),
                    "elapsed_h": round((time.time() - started) / 3600, 3),
                }
                with progress.open("a") as handle:
                    handle.write(json.dumps(entry) + "\n")
                print(json.dumps(entry), flush=True)
            del stats
            torch.cuda.empty_cache()
        return True

    passes: list[tuple[Path, Path | None]] = [(args.out, None)]
    for item in args.survivors:
        keep, _, table = item.partition("=")
        passes.append((args.out / f"student-{int(keep)}", Path(table)))
    for out_dir, table in passes:
        if rank == 0:
            print(f"pass -> {out_dir}", flush=True)
        if not run_pass(out_dir, mask_router(table)):
            break
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
