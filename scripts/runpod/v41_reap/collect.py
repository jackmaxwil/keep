"""REAP, imatrix, routes and teacher logits for DeepSeek-V4.1-Flash, on one GPU.

DeepSeek's own reference code (``inference/model.py`` at the pinned revision)
runs one layer at a time through ``stream.LayerStreamer``. Weights come straight
from the HF shards (no ``convert.py``), and any GPU of 80 GB or more works.
Design: ``docs/deepseek-v4-flash/2026-11-06-v41-reap-single-gpu-design.md``.

``MoE.forward`` is replaced by a copy that also records, for every routed slot,

- REAP statistics (``keep.quality.reap.ReapStats`` arrays, ``reap__*``);
- imatrix sums for the two expert input spaces (``hidden`` = the post
  ``ffn_norm`` row that w1/w3 read, ``down`` = the GLU activation w2 reads),
  under the keys ``keep.quality.dsv4_teacher_runner`` finalizes;
- the first ``ROUTE_RECORD_TOKENS`` routes of each session (``routes__*``),
  for ``benchmarks/replay_dsv4_expert_residency.py``.

Two sweeps over the weights:

- ``--sweep calibration`` writes REAP + imatrix + routes per calibration session.
- ``--sweep eval`` runs the teacher and every ``--survivors`` student side by
  side. The teacher writes REAP + routes and top-K logits per eval session. Each
  student is scored against the teacher's top-K (KL, NLL, top-1 agreement) and
  written to ``<out>/student-KEEP/``.

Output is one npz per session under ``<out>/<split>/sessions/``, so KEEP's
``finalize-reap``, the imatrix finalizers and ``score.py`` read it unchanged.

The replacement forward computes exactly what the reference does (same ops,
same order). ``tests/test_runpod_v41_reap.py`` checks it bit for bit against a
CPU copy of the reference Gate / Expert / MoE, and ``tests/test_runpod_v41_stream.py``
checks both sweeps against ``Transformer.forward``. One approximation: the
reference folds the gate weight in *before* w2 (``w2(g * h)``), so the expert
output norm ``||f||`` is taken as ``||w2(g * h)|| / g``. With FP8 activation
quantization in w2 that differs from ``||w2(h)||`` by rounding only.

Usage, on the pod, one process per GPU (``--shard I/N`` splits the sessions)::

    CUDA_VISIBLE_DEVICES=0 python3 collect.py --hf-path /workspace/v41-hf \\
        --pack /workspace/pack/v41-pack.jsonl --out /workspace/out --sweep calibration --shard 0/1

then finalize-reap and reap-select, then the same command with ``--sweep eval``
and ``--survivors 192=/workspace/out/survivors-192.json`` (repeatable).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
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
            for m in self.imatrix.values():
                m["total_route_count"][layer] += rows * top_k
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
            if space not in self.imatrix:
                continue
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
        for m in self.imatrix.values():
            for value in m.values():
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
        for space, m in self.imatrix.items():
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
# Sweeps (stream.LayerStreamer: one GPU, one layer at a time)
# ---------------------------------------------------------------------------

TEACHER = "teacher"


def chunk_sessions(sessions: list[tuple[str, list[dict[str, Any]]]], budget: int, streams: int) -> list[list[tuple[str, list[dict[str, Any]]]]]:
    """Whole sessions, in order, up to ``budget`` resident tokens (tokens x streams) per chunk."""

    chunks: list[list[tuple[str, list[dict[str, Any]]]]] = []
    size = 0
    for name, records in sessions:
        n = sum(len(r["token_ids"]) for r in records) * streams
        if not chunks or (chunks[-1] and size + n > budget):
            chunks.append([])
            size = 0
        chunks[-1].append((name, records))
        size += n
    return chunks


def survivor_masks(table: dict[str, Any], n_experts: int) -> dict[int, Any]:
    """reap-select table -> {layer: bool tensor, True = pruned}. Masking the router to -inf
    equals slicing it (tests/test_reap.py), so one set of weights scores every candidate."""

    import torch

    masks = {}
    for layer, kept in table["survivors"].items():
        mask = torch.ones(n_experts, dtype=torch.bool)
        mask[torch.as_tensor(kept)] = False
        masks[int(layer)] = mask
    return masks


def install_hooks(streamer: Any, *, top_k: int, rows_per_slice: int) -> dict[str, Any]:
    """Patch the reference MoE and the streamer's head once; returns the shared ``state``."""

    import torch
    import torch.nn.functional as F

    state: dict[str, Any] = {"rank": 0, "stats": None, "positions": None, "teacher": None}
    streamer.ref.MoE.forward = make_moe_forward(streamer.ref, torch, F, None, state)
    streamer.head.forward = make_head_forward(streamer.head, torch, F, None, state, top_k=top_k, rows_per_slice=rows_per_slice)
    return state


def run_chunk(
    streamer: Any,
    state: dict[str, Any],
    chunk: list[tuple[str, list[dict[str, Any]]]],
    *,
    split: str,
    students: dict[str, dict[int, Any]],
    dims: dict[str, int],
    device: Any,
) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    """One pass over the weights for ``chunk``; returns arrays per (stream, session).

    Calibration: the teacher only, REAP + imatrix + routes per session.
    Eval: the teacher (REAP + routes + top-K logits) and every student (scored against
    the teacher) side by side. Teacher records go through the MoE per session, so the
    stats stay per session; each student's records go through it together.
    """

    import torch
    from stream import Item

    args = streamer.args
    n_layers, n_experts = int(args.n_layers), int(args.n_routed_experts)
    stats = {
        name: SessionStats(torch=torch, n_layers=n_layers, n_experts=n_experts, dims=dims if split == "calibration" else {}, device=device)
        for name, _ in chunk
    }
    items: list[Item] = []
    streams_of: dict[str, list[tuple[str, Item]]] = {}
    for name, records in chunk:
        for record in records:
            tokens = torch.as_tensor([record["token_ids"]], dtype=torch.long, device=device)
            for stream in (TEACHER, *students) if split == "eval" else (TEACHER,):
                item = Item(tokens=tokens, group=(TEACHER, name) if stream == TEACHER else (stream, None))
                items.append(item)
                streams_of.setdefault(name, []).append((stream, item))

    def before_moe(layer: int, block: Any, key: tuple[str, str | None]) -> None:
        stream, name = key
        bias = block.ffn.gate.bias
        if not hasattr(block, "full_bias"):
            block.full_bias = bias.data.clone()
        bias.data.copy_(block.full_bias)
        mask = students[stream].get(layer) if stream != TEACHER else None
        if mask is not None:
            bias.data[mask.to(bias.device)] = float("-inf")
        state["stats"] = stats[name] if stream == TEACHER and layer < n_layers else None

    with torch.inference_mode():
        streamer.forward(items, before_moe)
    state["stats"] = None

    layers = list(range(n_layers))
    results: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for name, records in chunk:
        results[(TEACHER, name)] = stats[name].arrays(layers)
        if split != "eval":
            continue
        (record,) = records
        positions = target_positions(record)
        targets = np.asarray(record["token_ids"], dtype=np.int64)[positions + 1]
        state["positions"] = torch.as_tensor(positions, device=device)
        state["targets"] = torch.as_tensor(targets, device=device)
        base = {"positions": positions.astype(np.int32), "target_token_ids": targets.astype(np.int32)}
        state["teacher"] = None
        for stream, item in streams_of[name]:  # the teacher first, then the students it scores
            state["captured"] = {k: [] for k in CAPTURE_KEYS}
            with torch.inference_mode():
                streamer.head(streamer.final_hidden(item))
            cap = {k: np.concatenate(v) for k, v in state["captured"].items() if v}
            results.setdefault((stream, name), {}).update(base | cap)
            if stream == TEACHER:
                state["teacher"] = {  # the fp16 values the npz keeps, as the old two-pass flow read them
                    "ids": torch.as_tensor(cap["topk_logit_ids"].astype(np.int64), device=device),
                    "logprob": torch.as_tensor(cap["topk_logit_values"].astype(np.float32) - cap["logsumexp"][:, None], device=device),
                    "targets": state["targets"],
                }
        state["positions"] = state["targets"] = state["teacher"] = None
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hf-path", required=True, type=Path, help="the pinned HF checkpoint (safetensors shards + tokenizer)")
    parser.add_argument("--ref-dir", type=Path, help="reference inference/ directory (default: HF_PATH/inference)")
    parser.add_argument("--pack", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--sweep", required=True, choices=("calibration", "eval"))
    parser.add_argument(
        "--survivors", action="append", default=[], metavar="KEEP=PATH",
        help="eval sweep (repeatable): also run a student with the router masked to this reap-select "
        "table, scored against the teacher, into <out>/student-KEEP/",
    )
    parser.add_argument("--chunk-tokens", type=int, default=524288, help="resident tokens per weight pass (x streams)")
    parser.add_argument("--shard", default="0/1", help="I/N: this process takes every Nth session from I")
    parser.add_argument("--max-seq-len", type=int, default=16384)
    parser.add_argument("--top-k", type=int, default=2048, help="teacher logits kept per eval position")
    parser.add_argument("--rows-per-slice", type=int, default=256)
    parser.add_argument("--deadline-utc", help="stop cleanly before the first chunk that starts after this time")
    parser.add_argument("--skip-hash-check", action="store_true", help="tests only")
    args = parser.parse_args(argv)
    if args.survivors and args.sweep != "eval":
        parser.error("--survivors needs --sweep eval")
    shard, n_shards = map(int, args.shard.split("/"))

    import torch
    from transformers import AutoTokenizer

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import stream

    ref_dir = args.ref_dir or args.hf_path / "inference"
    digest = hashlib.sha256((ref_dir / "model.py").read_bytes()).hexdigest()
    if not args.skip_hash_check and digest != REFERENCE_MODEL_PY_SHA256:
        raise SystemExit(f"{ref_dir / 'model.py'} sha256 {digest} is not the pinned reference ({REFERENCE_MODEL_PY_SHA256})")
    ref = stream.import_reference(ref_dir)

    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.manual_seed(33377335)
    model_args = ref.ModelArgs(**json.loads((ref_dir / "config.json").read_text()))
    model_args.max_batch_size = 1
    model_args.max_seq_len = args.max_seq_len

    ckpt = stream.HFCheckpoint(args.hf_path)
    layout = ref.EngramLayout.from_args(model_args)
    hash_state = None
    if layout is not None:
        with torch.device("cuda"):
            hash_state = ref.NgramHashState(model_args, layout, AutoTokenizer.from_pretrained(args.hf_path))
    engram_rows = {layer: stream.EngramRows.from_checkpoint(ckpt, layer, ref.fp8_block_size) for layer in model_args.engram_layer_ids}
    streamer = stream.LayerStreamer(ref, model_args, ckpt.load, engram_rows, hash_state, "cuda")
    state = install_hooks(streamer, top_k=args.top_k, rows_per_slice=args.rows_per_slice)

    n_experts = int(model_args.n_routed_experts)
    students: dict[str, dict[int, Any]] = {}
    survivors_sha: dict[str, str] = {}
    for spec in args.survivors:
        keep, _, table = spec.partition("=")
        students[keep] = survivor_masks(json.loads(Path(table).read_text()), n_experts)
        survivors_sha[keep] = hashlib.sha256(Path(table).read_bytes()).hexdigest()

    run_config = {
        "model_id": MODEL_ID,
        "revision": PINNED_REVISION,
        "model_py_sha256": digest,
        "pack_sha256": hashlib.sha256(args.pack.read_bytes()).hexdigest(),
        "max_seq_len": args.max_seq_len,
        "top_k": args.top_k,
        "driver": "layer-streamed, world_size 1",
    }
    text = json.dumps(run_config, indent=2, sort_keys=True) + "\n"
    config_path = args.out / "run-config.json"
    args.out.mkdir(parents=True, exist_ok=True)
    if not config_path.exists() or config_path.read_text() != text:
        config_path.write_text(text)
    config_sha = hashlib.sha256(json.dumps(run_config, sort_keys=True).encode()).hexdigest()
    teacher_config_sha = hashlib.sha256(text.encode()).hexdigest()

    def target(stream_name: str, name: str) -> Path:
        root = args.out if stream_name == TEACHER else args.out / f"student-{stream_name}"
        return root / args.sweep / "sessions" / f"{name}.npz"

    streams = (TEACHER, *students)
    todo = [
        (name, records)
        for i, (name, records) in enumerate(item for item in load_sessions(args.pack).items() if item[1][0]["split"] == args.sweep)
        if i % n_shards == shard and not all(target(s, name).exists() for s in streams)
    ]
    for name, records in todo:
        if any(len(r["token_ids"]) > args.max_seq_len for r in records):
            raise ValueError(f"session {name} has a record over --max-seq-len {args.max_seq_len}")
    deadline = None
    if args.deadline_utc:
        deadline = dt.datetime.fromisoformat(args.deadline_utc.replace("Z", "+00:00")).timestamp()
    dims = {"hidden": int(model_args.dim), "down": int(model_args.moe_inter_dim)}
    started = time.time()
    progress = args.out / f"progress-{args.sweep}-{shard}of{n_shards}.jsonl"
    for chunk in chunk_sessions(todo, args.chunk_tokens, len(streams)):
        if deadline is not None and time.time() > deadline:
            print("deadline reached; stopping cleanly", flush=True)
            break
        t0 = time.time()
        results = run_chunk(streamer, state, chunk, split=args.sweep, students=students, dims=dims, device="cuda")
        for name, records in chunk:
            ident = identity(session=name, split=args.sweep, records=records, config_sha=config_sha, chunk=args.max_seq_len)
            for stream_name in streams:
                arrays = ident | results[(stream_name, name)]
                if stream_name != TEACHER:
                    arrays["record_type"] = np.asarray("dsv4_masked_student_eval_v1")
                    arrays["survivors_sha256"] = np.asarray(survivors_sha[stream_name])
                    arrays["teacher_run_config_sha256"] = np.asarray(teacher_config_sha)
                save_npz(target(stream_name, name), arrays)
        tokens = sum(len(r["token_ids"]) for _, records in chunk for r in records)
        entry = {
            "sessions": [name for name, _ in chunk],
            "tokens": tokens,
            "streams": len(streams),
            "seconds": round(time.time() - t0, 3),
            "tok_per_s": round(tokens * len(streams) / max(time.time() - t0, 1e-9), 1),
            "max_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
            "elapsed_h": round((time.time() - started) / 3600, 3),
        }
        with progress.open("a") as handle:
            handle.write(json.dumps(entry) + "\n")
        print(json.dumps(entry), flush=True)
        del results
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
