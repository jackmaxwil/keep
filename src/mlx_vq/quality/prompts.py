from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Literal


@dataclass(frozen=True)
class QualityPrompt:
    prompt_id: str
    text: str
    max_new_tokens: int
    context_tokens: int | None = None
    eval_split: Literal["selection", "report", "holdout"] | None = None
    suite_version: str | None = None
    row_intent: str | None = None


_LADDER_SUITE_VERSION = "air_vq_ladder_v1"
_IMATRIX_SUITE_VERSION = "air_imatrix_calib_v1"


_LONG_RECALL_SENTENCE = (
    "GLM-4.5-Air quality baseline document. The key city is Lyon, the key color is teal, "
    "and the key number is 42. "
)

_LONG_RECALL_SHAPE_SENTENCE = (
    "GLM-4.5-Air widened evaluation document. The key city is Porto, "
    "the key shape is triangle, and the key code is R7. "
)


def _long_recall_prompt() -> str:
    document = (_LONG_RECALL_SENTENCE * 80).strip()
    return f"{document}\nQuestion: What is the key number? Answer with only the number."


def _long_recall_shape_prompt() -> str:
    document = (_LONG_RECALL_SHAPE_SENTENCE * 80).strip()
    return f"{document}\nQuestion: What is the key shape? Answer with only the shape."


_BASE_QUALITY_PROMPTS: tuple[QualityPrompt, ...] = (
    QualityPrompt(
        prompt_id="capital_france",
        text="The capital of France is",
        max_new_tokens=8,
    ),
    QualityPrompt(
        prompt_id="short_math",
        text="Compute 17 + 25. Answer with only the number.",
        max_new_tokens=8,
    ),
    QualityPrompt(
        prompt_id="code_completion",
        text="def fibonacci(n):",
        max_new_tokens=24,
    ),
    QualityPrompt(
        prompt_id="instruction_following",
        text="Write one sentence about why caches help inference.",
        max_new_tokens=24,
    ),
    QualityPrompt(
        prompt_id="long_recall_1k",
        text=_long_recall_prompt(),
        max_new_tokens=8,
        context_tokens=1024,
    ),
)

_LANE0_WIDENED_EXTRA_PROMPTS: tuple[QualityPrompt, ...] = (
    QualityPrompt(
        prompt_id="short_math_addition_carry",
        text="Compute 68 + 57. Answer with only the number.",
        max_new_tokens=8,
    ),
    QualityPrompt(
        prompt_id="short_math_multi_step",
        text="Compute (14 * 3) - 9. Answer with only the number.",
        max_new_tokens=8,
    ),
    QualityPrompt(
        prompt_id="short_math_half",
        text="What is half of 38? Answer with only the number.",
        max_new_tokens=8,
    ),
    QualityPrompt(
        prompt_id="code_completion_loop",
        text="def count_vowels(text):",
        max_new_tokens=24,
    ),
    QualityPrompt(
        prompt_id="code_completion_branch",
        text="def classify_temperature(celsius):",
        max_new_tokens=24,
    ),
    QualityPrompt(
        prompt_id="code_completion_list",
        text="def unique_in_order(items):",
        max_new_tokens=24,
    ),
    QualityPrompt(
        prompt_id="instruction_following_six_words",
        text="In exactly six words, explain why batching speeds inference.",
        max_new_tokens=16,
    ),
    QualityPrompt(
        prompt_id="instruction_following_json",
        text='Return compact JSON with keys "tool" and "benefit" explaining caching.',
        max_new_tokens=32,
    ),
    QualityPrompt(
        prompt_id="instruction_following_avoid_word",
        text="Answer in one sentence without using the word fast: why do indexes help search?",
        max_new_tokens=24,
    ),
    QualityPrompt(
        prompt_id="capital_japan",
        text="The capital of Japan is",
        max_new_tokens=8,
    ),
    QualityPrompt(
        prompt_id="basic_color",
        text="Grass is usually the color",
        max_new_tokens=8,
    ),
    QualityPrompt(
        prompt_id="long_recall_shape_1k",
        text=_long_recall_shape_prompt(),
        max_new_tokens=8,
        context_tokens=1024,
    ),
)


def _ladder_prompt(
    *,
    split: Literal["selection", "report", "holdout"],
    category: str,
    index: int,
    text: str,
    max_new_tokens: int,
    context_tokens: int | None = None,
    row_intent: str | None = None,
) -> QualityPrompt:
    prefix = {"selection": "select", "report": "report", "holdout": "holdout"}[split]
    return QualityPrompt(
        prompt_id=f"{prefix}_{category}_{index:03d}",
        text=text,
        max_new_tokens=max_new_tokens,
        context_tokens=context_tokens,
        eval_split=split,
        suite_version=_LADDER_SUITE_VERSION,
        row_intent=row_intent or category,
    )


def _make_code_debug_prompts(
    split: Literal["selection", "report", "holdout"], *, offset: int
) -> tuple[QualityPrompt, ...]:
    templates = (
        "def parse_record_{n}(row):\n    \"\"\"Return the normalized id and value.\"\"\"",
        "def repair_cache_entry_{n}(entry):\n    if entry is None:",
        "def route_tokens_{n}(scores, top_k):\n    # Return selected expert ids in rank order.",
        "def summarize_errors_{n}(items):\n    totals = {{}}",
        "def stable_partition_{n}(values, predicate):\n    left = []",
        "def validate_tensor_shape_{n}(shape):\n    expected_rank = 2",
        "def dedupe_prompts_{n}(prompts):\n    seen = set()",
        "def compute_tail_metric_{n}(rows):\n    values = []",
        "def format_gate_report_{n}(verdict):\n    reasons = verdict.get('reasons', [])",
        "def load_jsonl_rows_{n}(path):\n    rows = []",
        "def clamp_router_temperature_{n}(value):\n    return",
    )
    prompts = []
    for idx in range(44):
        n = offset + idx
        prompts.append(
            _ladder_prompt(
                split=split,
                category="code",
                index=idx,
                text=templates[idx % len(templates)].format(n=n),
                max_new_tokens=32,
                row_intent="code_debug",
            )
        )
    return tuple(prompts)


def _make_arithmetic_prompts(
    split: Literal["selection", "report", "holdout"], *, offset: int
) -> tuple[QualityPrompt, ...]:
    prompts = []
    for idx in range(28):
        a = 37 + offset + idx * 3
        b = 19 + (offset // 2) + idx * 2
        if idx % 4 == 0:
            text = f"Compute ({a} + {b}) - {idx + 5}. Answer with only the number."
        elif idx % 4 == 1:
            text = f"Compute {a} * 3 + {b}. Answer with only the number."
        elif idx % 4 == 2:
            text = f"What is {a + b} divided by 2? Answer with only the number."
        else:
            text = f"Compute {a + 11} - {b}. Answer with only the number."
        prompts.append(
            _ladder_prompt(
                split=split,
                category="math",
                index=idx,
                text=text,
                max_new_tokens=8,
                row_intent="hard_arithmetic",
            )
        )
    return tuple(prompts)


def _make_instruction_prompts(
    split: Literal["selection", "report", "holdout"], *, offset: int
) -> tuple[QualityPrompt, ...]:
    topics = (
        "cache warming",
        "router calibration",
        "tensor validation",
        "prompt splitting",
        "checkpoint backups",
        "memory counters",
        "tail metrics",
        "expert coverage",
        "distributed preflight",
        "artifact manifests",
    )
    prompts = []
    for idx in range(20):
        topic = topics[idx % len(topics)]
        n = offset + idx
        if idx % 2 == 0:
            text = (
                f"In exactly {6 + (idx % 4)} words, explain why {topic} "
                f"matters for case {n}."
            )
        else:
            text = (
                f"Return compact JSON with keys \"step\" and \"reason\" for "
                f"{topic} case {n}."
            )
        prompts.append(
            _ladder_prompt(
                split=split,
                category="instruction",
                index=idx,
                text=text,
                max_new_tokens=32,
                row_intent="instruction_following",
            )
        )
    return tuple(prompts)


def _make_long_recall_prompts(
    split: Literal["selection", "report", "holdout"], *, offset: int
) -> tuple[QualityPrompt, ...]:
    prompts = []
    for idx in range(16):
        city = ("Lyon", "Porto", "Oslo", "Quito", "Busan", "Riga", "Turin", "Nara")[idx % 8]
        color = ("teal", "amber", "violet", "silver", "green", "navy", "coral", "black")[idx % 8]
        number = 300 + offset + idx
        sentence = (
            f"Ladder memo {offset + idx}: the city is {city}, the color is {color}, "
            f"and the checksum number is {number}. "
        )
        text = (sentence * 64).strip()
        if idx % 3 == 0:
            question = "Question: What is the checksum number? Answer with only the number."
        elif idx % 3 == 1:
            question = "Question: What is the city? Answer with only the city."
        else:
            question = "Question: What is the color? Answer with only the color."
        prompts.append(
            _ladder_prompt(
                split=split,
                category="long",
                index=idx,
                text=f"{text}\n{question}",
                max_new_tokens=8,
                context_tokens=1024,
                row_intent="long_recall",
            )
        )
    return tuple(prompts)


def _make_control_prompts(
    split: Literal["selection", "report", "holdout"], *, offset: int
) -> tuple[QualityPrompt, ...]:
    facts = (
        ("France", "Paris"),
        ("Japan", "Tokyo"),
        ("Italy", "Rome"),
        ("Canada", "Ottawa"),
        ("triangle", "three"),
        ("week", "seven"),
    )
    prompts = []
    for idx in range(12):
        key, answer = facts[idx % len(facts)]
        text = (
            f"Control item {offset + idx}: for {key}, the expected short answer is "
            f"{answer}. Repeat only the expected short answer."
        )
        prompts.append(
            _ladder_prompt(
                split=split,
                category="control",
                index=idx,
                text=text,
                max_new_tokens=8,
                row_intent="robust_control",
            )
        )
    return tuple(prompts)


def _make_route_prompts(
    split: Literal["selection", "report", "holdout"], *, offset: int
) -> tuple[QualityPrompt, ...]:
    prompts = []
    for idx in range(8):
        n = offset + idx
        text = (
            f"Debug the GLU route case {n}. Given scores for experts 31, 36, and 41, "
            "explain in code comments how to keep layer 41 code-completion routes "
            "covered without changing router weights."
        )
        prompts.append(
            _ladder_prompt(
                split=split,
                category="route",
                index=idx,
                text=text,
                max_new_tokens=40,
                row_intent="layer41_glu_code_route",
            )
        )
    return tuple(prompts)


def _make_air_vq_ladder_prompts(
    split: Literal["selection", "report", "holdout"],
    *,
    offset: int,
) -> tuple[QualityPrompt, ...]:
    prompts = (
        _make_code_debug_prompts(split, offset=offset)
        + _make_arithmetic_prompts(split, offset=offset)
        + _make_instruction_prompts(split, offset=offset)
        + _make_long_recall_prompts(split, offset=offset)
        + _make_control_prompts(split, offset=offset)
        + _make_route_prompts(split, offset=offset)
    )
    if len(prompts) != 128:
        raise RuntimeError(f"expected 128 {split} prompts, got {len(prompts)}")
    return prompts


_AIR_VQ_LADDER_SELECT_V1_PROMPTS = _make_air_vq_ladder_prompts(
    "selection",
    offset=0,
)
_AIR_VQ_LADDER_REPORT_V1_PROMPTS = _make_air_vq_ladder_prompts(
    "report",
    offset=1000,
)
_AIR_VQ_LADDER_HOLDOUT_V1_PROMPTS = _make_air_vq_ladder_prompts(
    "holdout",
    offset=2000,
)


def _glm45_air_chat_prompt(*, system: str, user: str) -> str:
    return (
        f"<|system|>\n{system.strip()}\n"
        f"<|user|>\n{user.strip()}\n"
        "<|assistant|>\n"
    )


def _imatrix_prompt(
    *,
    category: str,
    index: int,
    user: str,
    row_intent: str,
    max_new_tokens: int,
    context_tokens: int | None = None,
) -> QualityPrompt:
    return QualityPrompt(
        prompt_id=f"imatrix_calib_{category}_{index:03d}",
        text=_glm45_air_chat_prompt(
            system=(
                "You are GLM-4.5-Air in calibration mode. Follow the user task "
                "literally, preserve code syntax, and answer with the requested artifact."
            ),
            user=user,
        ),
        max_new_tokens=max_new_tokens,
        context_tokens=context_tokens,
        suite_version=_IMATRIX_SUITE_VERSION,
        row_intent=row_intent,
    )


def _make_imatrix_code_debug_prompts() -> tuple[QualityPrompt, ...]:
    templates = (
        "Inspect this Python cache loader and return the smallest patch plus one regression test:\n"
        "def load_rows_{n}(path):\n    rows = []\n    for line in path.read_text().splitlines():\n        if line:\n            rows.append(json.loads(line))\n    return rows",
        "Debug this tensor-shape validator and explain the failing edge case in comments:\n"
        "def validate_shape_{n}(shape, expected):\n    if len(shape) == len(expected):\n        return True\n    return all(a == b for a, b in zip(shape, expected))",
        "Repair this route aggregation helper without changing the public return schema:\n"
        "def summarize_routes_{n}(records):\n    counts = {{}}\n    for row in records:\n        for expert in row['experts']:\n            counts[expert] = 1\n    return counts",
        "Find the numerical bug in this quantization metric helper and write the corrected function:\n"
        "def mean_kld_{n}(rows):\n    return sum(row['kld'] for row in rows) / len(rows or [1])",
        "Review this safetensors manifest update and return a concise patch plan:\n"
        "manifest['groups'][name]['ready'] = True\nmanifest['ready_count'] += len(manifest['groups'])",
    )
    return tuple(
        _imatrix_prompt(
            category="code_debug",
            index=idx,
            user=(
                f"Calibration code-debug case {700 + idx}. "
                + templates[idx % len(templates)].format(n=700 + idx)
            ),
            row_intent="imatrix_code_debug",
            max_new_tokens=96,
        )
        for idx in range(20)
    )


def _make_imatrix_code_completion_prompts() -> tuple[QualityPrompt, ...]:
    templates = (
        "Complete the function with robust error handling and no external dependencies:\n"
        "def merge_prompt_hashes_{n}(selection_rows, report_rows):",
        "Complete this deterministic planner; preserve input order for ties:\n"
        "def choose_precision_tiers_{n}(scores, budget_bits):",
        "Finish this parser so malformed rows are skipped with a reason string:\n"
        "def parse_teacher_cache_row_{n}(line):",
        "Complete this activation-stat accumulator using numerically stable updates:\n"
        "def update_channel_importance_{n}(state, activations):",
        "Write the body of this CLI helper and include the JSON object it should emit:\n"
        "def build_imatrix_manifest_{n}(output_dir, prompt_set, layer_count):",
    )
    return tuple(
        _imatrix_prompt(
            category="code_completion",
            index=idx,
            user=templates[idx % len(templates)].format(n=800 + idx),
            row_intent="imatrix_code_completion",
            max_new_tokens=128,
        )
        for idx in range(20)
    )


def _make_imatrix_code_test_prompts() -> tuple[QualityPrompt, ...]:
    templates = (
        "Write pytest coverage for deterministic prompt dedupe. Include one no-overlap case and one collision case.",
        "Write tests for an allocator that must keep protected layers at or above their floor tier.",
        "Write a small synthetic tensor test where importance_j equals the sum of squared activation column j.",
        "Write regression tests for a loader that must behave byte-for-byte the same when a tier map is absent.",
        "Write tests for a report summarizer that ranks mean KLD before perplexity and preserves p999 KLD.",
    )
    return tuple(
        _imatrix_prompt(
            category="code_tests",
            index=idx,
            user=f"Calibration test-design case {900 + idx}. {templates[idx % len(templates)]}",
            row_intent="imatrix_code_tests",
            max_new_tokens=128,
        )
        for idx in range(20)
    )


def _make_imatrix_code_algorithm_prompts() -> tuple[QualityPrompt, ...]:
    templates = (
        "Design pseudocode for a greedy bit allocator that spends the next bit where importance-weighted error drops most.",
        "Explain how to compute per-column activation importance for a routed expert projection without forcing cold experts.",
        "Given three candidate tiers, describe how to estimate error reduction per byte for one projection shard.",
        "Sketch a deterministic manifest schema for per-layer per-projection imatrix sidecars and coverage metadata.",
    )
    return tuple(
        _imatrix_prompt(
            category="code_algorithm",
            index=idx,
            user=f"Imatrix algorithm calibration case {1000 + idx}. {templates[idx % len(templates)]}",
            row_intent="imatrix_code_algorithm",
            max_new_tokens=128,
        )
        for idx in range(4)
    )


def _make_imatrix_reasoning_prompts() -> tuple[QualityPrompt, ...]:
    prompts = []
    for idx in range(12):
        a = 41 + idx * 7
        b = 13 + idx * 5
        prompts.append(
            _imatrix_prompt(
                category="reasoning",
                index=idx,
                user=(
                    f"Solve this calibration arithmetic case step by step, then give only the final integer: "
                    f"(({a} * 3) + {b}) - {idx + 11}."
                ),
                row_intent="imatrix_reasoning_math",
                max_new_tokens=48,
            )
        )
    return tuple(prompts)


def _make_imatrix_instruction_prompts() -> tuple[QualityPrompt, ...]:
    topics = (
        "activation coverage",
        "teacher-cache validation",
        "route-trace leakage",
        "artifact manifests",
        "KLD tail metrics",
        "single-host memory gates",
    )
    prompts = []
    for idx in range(12):
        topic = topics[idx % len(topics)]
        if idx % 2 == 0:
            user = (
                f"Instruction calibration case {1200 + idx}. In exactly {7 + idx % 4} words, "
                f"explain why {topic} matters for quantization calibration."
            )
        else:
            user = (
                f"Instruction calibration case {1200 + idx}. Return compact JSON with keys step "
                f"and reason for checking {topic} before materialization."
            )
        prompts.append(
            _imatrix_prompt(
                category="instruction",
                index=idx,
                user=user,
                row_intent="imatrix_instruction_following",
                max_new_tokens=64,
            )
        )
    return tuple(prompts)


def _make_imatrix_long_recall_prompts() -> tuple[QualityPrompt, ...]:
    prompts = []
    for idx in range(8):
        project = ("Atlas", "Boreal", "Cipher", "Delta")[idx % 4]
        metric = ("mean KLD", "top-1 agreement", "p999 KLD", "effective bits")[idx % 4]
        code = 5000 + idx * 37
        sentence = (
            f"Calibration memo {idx}: project {project} tracks {metric}; "
            f"the routing checksum is {code}; the protected projection is gate/up. "
        )
        document = (sentence * 24).strip()
        prompts.append(
            _imatrix_prompt(
                category="long_recall",
                index=idx,
                user=(
                    f"{document}\nQuestion: What is the routing checksum for calibration memo {idx}? "
                    "Answer with only the number."
                ),
                row_intent="imatrix_long_recall",
                max_new_tokens=8,
                context_tokens=768,
            )
        )
    return tuple(prompts)


_AIR_IMATRIX_CALIB_V1_PROMPTS = (
    _make_imatrix_code_debug_prompts()
    + _make_imatrix_code_completion_prompts()
    + _make_imatrix_code_test_prompts()
    + _make_imatrix_code_algorithm_prompts()
    + _make_imatrix_reasoning_prompts()
    + _make_imatrix_instruction_prompts()
    + _make_imatrix_long_recall_prompts()
)
if len(_AIR_IMATRIX_CALIB_V1_PROMPTS) != 96:
    raise RuntimeError(f"expected 96 imatrix calibration prompts, got {len(_AIR_IMATRIX_CALIB_V1_PROMPTS)}")

def _load_instruction_hf_v1_prompts() -> tuple[QualityPrompt, ...]:
    """Load the disjoint HF instruction-following prompts (for Phase A recovery).

    Read at import from a repo-relative JSONL ({prompt_id, text, category}). These are
    TRAINING prompts for KD recovery of the weak instruction domain — deliberately
    disjoint from the ladder eval splits (no eval_split tag). Absent file -> empty set.
    """
    import json
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[3] / "artifacts/quality/instruction_hf_dolly48_prompts.jsonl"
    if not path.exists():
        return ()
    loaded: list[QualityPrompt] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        loaded.append(
            QualityPrompt(
                prompt_id=str(row["prompt_id"]),
                text=str(row["text"]),
                max_new_tokens=32,
                suite_version="instruction_hf_v1",
                row_intent=str(row.get("category", "instruction_following")),
            )
        )
    return tuple(loaded)


_INSTRUCTION_HF_V1_PROMPTS = _load_instruction_hf_v1_prompts()


_PROMPT_SETS: dict[str, tuple[QualityPrompt, ...]] = {
    "base": _BASE_QUALITY_PROMPTS,
    "lane0_widened": _BASE_QUALITY_PROMPTS + _LANE0_WIDENED_EXTRA_PROMPTS,
    "air_vq_ladder_select_v1": _AIR_VQ_LADDER_SELECT_V1_PROMPTS,
    "air_vq_ladder_report_v1": _AIR_VQ_LADDER_REPORT_V1_PROMPTS,
    "air_vq_ladder_holdout_v1": _AIR_VQ_LADDER_HOLDOUT_V1_PROMPTS,
    "air_imatrix_calib_v1": _AIR_IMATRIX_CALIB_V1_PROMPTS,
    "instruction_hf_v1": _INSTRUCTION_HF_V1_PROMPTS,
}

_PROMPTS_BY_ID: dict[str, QualityPrompt] = {
    prompt.prompt_id: prompt
    for prompts in _PROMPT_SETS.values()
    for prompt in prompts
}


def normalize_prompt_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def prompt_text_hash(text: str) -> str:
    return hashlib.sha256(normalize_prompt_text(text).encode("utf-8")).hexdigest()


def quality_prompt_metadata(prompt: QualityPrompt, *, prompt_set: str) -> dict[str, str]:
    metadata = {
        "prompt_set": prompt_set,
        "prompt_text_sha256": prompt_text_hash(prompt.text),
    }
    if prompt.eval_split is not None:
        metadata["eval_split"] = prompt.eval_split
    if prompt.suite_version is not None:
        metadata["suite_version"] = prompt.suite_version
    if prompt.row_intent is not None:
        metadata["row_intent"] = prompt.row_intent
    return metadata


def get_quality_prompt_set_names() -> tuple[str, ...]:
    return tuple(_PROMPT_SETS)


def get_quality_prompt_ids(*, prompt_set: str | None = "base") -> tuple[str, ...]:
    if prompt_set is None:
        return tuple(sorted(_PROMPTS_BY_ID))
    return tuple(prompt.prompt_id for prompt in get_quality_prompts(prompt_set=prompt_set))


def get_quality_prompt_by_id(prompt_id: str) -> QualityPrompt:
    try:
        return _PROMPTS_BY_ID[prompt_id]
    except KeyError as error:
        choices = ", ".join(get_quality_prompt_ids(prompt_set=None))
        raise ValueError(f"unknown quality prompt {prompt_id!r}; choices: {choices}") from error


def get_quality_prompts(prompt_set: str = "base") -> tuple[QualityPrompt, ...]:
    try:
        prompts = _PROMPT_SETS[prompt_set]
    except KeyError as error:
        choices = ", ".join(get_quality_prompt_set_names())
        raise ValueError(f"unknown quality prompt set {prompt_set!r}; choices: {choices}") from error
    return prompts


def get_lane0_widened_quality_prompts() -> tuple[QualityPrompt, ...]:
    return get_quality_prompts(prompt_set="lane0_widened")


def _estimated_text_tokens(text: str) -> int:
    return len(re.findall(r"\S+", text))


def summarize_quality_prompt_corpus(prompt_set: str) -> dict[str, object]:
    prompts = get_quality_prompts(prompt_set=prompt_set)
    row_intent_counts: dict[str, int] = {}
    token_counts: dict[str, int] = {}
    for prompt in prompts:
        intent = prompt.row_intent or "unspecified"
        row_intent_counts[intent] = row_intent_counts.get(intent, 0) + 1
        token_counts[prompt.prompt_id] = _estimated_text_tokens(prompt.text)
    code_count = sum(
        count
        for intent, count in row_intent_counts.items()
        if intent.startswith("imatrix_code")
    )
    if prompt_set == "air_imatrix_calib_v1":
        compared_sets = (
            "air_vq_ladder_select_v1",
            "air_vq_ladder_report_v1",
            "air_vq_ladder_holdout_v1",
        )
    elif prompt_set.startswith("air_vq_ladder_"):
        compared_sets = tuple(
            name
            for name in (
                "air_vq_ladder_select_v1",
                "air_vq_ladder_report_v1",
                "air_vq_ladder_holdout_v1",
            )
            if name != prompt_set
        )
    else:
        compared_sets = ()
    return {
        "schema_version": 1,
        "prompt_set": prompt_set,
        "prompt_count": len(prompts),
        "row_intent_counts": row_intent_counts,
        "estimated_text_token_count": sum(token_counts.values()),
        "estimated_text_token_min": min(token_counts.values()) if token_counts else 0,
        "estimated_text_token_max": max(token_counts.values()) if token_counts else 0,
        "estimated_text_tokens_by_prompt": token_counts,
        "context_token_count": sum(int(prompt.context_tokens or 0) for prompt in prompts),
        "code_prompt_fraction": 0.0 if not prompts else code_count / len(prompts),
        "dedupe_checked_against": list(compared_sets),
    }


def validate_imatrix_calibration_prompt_set(
    *,
    compared_prompt_sets: tuple[str, ...] = (
        "air_vq_ladder_select_v1",
        "air_vq_ladder_report_v1",
        "air_vq_ladder_holdout_v1",
    ),
) -> dict[str, object]:
    prompts = get_quality_prompts(prompt_set="air_imatrix_calib_v1")
    errors: list[str] = []
    overlap_prompt_pairs: list[str] = []
    normalized_by_text: dict[str, str] = {}
    hashes_by_text: dict[str, str] = {}

    for prompt in prompts:
        if not prompt.prompt_id.startswith("imatrix_calib_"):
            errors.append(f"{prompt.prompt_id} missing imatrix_calib_ prefix")
        if prompt.eval_split is not None:
            errors.append(f"{prompt.prompt_id} should not belong to selection/report split")
        if prompt.suite_version != _IMATRIX_SUITE_VERSION:
            errors.append(f"{prompt.prompt_id} has suite_version={prompt.suite_version!r}")
        if not prompt.row_intent or not prompt.row_intent.startswith("imatrix_"):
            errors.append(f"{prompt.prompt_id} has invalid row_intent={prompt.row_intent!r}")
        if not all(marker in prompt.text for marker in ("<|system|>", "<|user|>", "<|assistant|>")):
            errors.append(f"{prompt.prompt_id} is not GLM chat-template shaped")

        normalized = normalize_prompt_text(prompt.text)
        if normalized in normalized_by_text:
            errors.append(
                "duplicate normalized imatrix prompt text: "
                f"{normalized_by_text[normalized]} and {prompt.prompt_id}"
            )
        normalized_by_text[normalized] = prompt.prompt_id
        text_hash = prompt_text_hash(prompt.text)
        if text_hash in hashes_by_text:
            errors.append(
                "duplicate imatrix prompt text hash: "
                f"{hashes_by_text[text_hash]} and {prompt.prompt_id}"
            )
        hashes_by_text[text_hash] = prompt.prompt_id

    for compared_set in compared_prompt_sets:
        for other in get_quality_prompts(prompt_set=compared_set):
            normalized = normalize_prompt_text(other.text)
            text_hash = prompt_text_hash(other.text)
            calib_id = normalized_by_text.get(normalized) or hashes_by_text.get(text_hash)
            if calib_id is not None:
                overlap_prompt_pairs.append(f"{calib_id}:{compared_set}:{other.prompt_id}")
    if overlap_prompt_pairs:
        errors.append("imatrix calibration prompt text overlaps an authority eval split")

    coverage = summarize_quality_prompt_corpus("air_imatrix_calib_v1")
    if float(coverage["code_prompt_fraction"]) < 0.60:
        errors.append("imatrix calibration prompt set is not code-heavy")

    return {
        "ok": not errors,
        "prompt_set": "air_imatrix_calib_v1",
        "prompt_count": len(prompts),
        "compared_prompt_sets": list(compared_prompt_sets),
        "overlap_prompt_pairs": overlap_prompt_pairs,
        "coverage": coverage,
        "errors": errors,
    }


def validate_quality_prompt_splits() -> dict[str, object]:
    split_sets = {
        "selection": get_quality_prompts(prompt_set="air_vq_ladder_select_v1"),
        "report": get_quality_prompts(prompt_set="air_vq_ladder_report_v1"),
        "holdout": get_quality_prompts(prompt_set="air_vq_ladder_holdout_v1"),
    }
    errors: list[str] = []
    duplicate_normalized: list[str] = []
    normalized_by_split: dict[str, dict[str, str]] = {}
    for split, prompts in split_sets.items():
        by_text: dict[str, str] = {}
        for prompt in prompts:
            if prompt.eval_split != split:
                errors.append(f"{prompt.prompt_id} has eval_split={prompt.eval_split!r}")
            if prompt.suite_version != _LADDER_SUITE_VERSION:
                errors.append(f"{prompt.prompt_id} has suite_version={prompt.suite_version!r}")
            normalized = normalize_prompt_text(prompt.text)
            if normalized in by_text:
                errors.append(
                    f"duplicate normalized prompt text in {split}: "
                    f"{by_text[normalized]} and {prompt.prompt_id}"
                )
            by_text[normalized] = prompt.prompt_id
        normalized_by_split[split] = by_text
    split_names = tuple(split_sets)
    for left_index, left_split in enumerate(split_names):
        for right_split in split_names[left_index + 1 :]:
            for normalized, left_id in normalized_by_split[left_split].items():
                right_id = normalized_by_split[right_split].get(normalized)
                if right_id is not None:
                    duplicate_normalized.append(f"{left_id}:{right_id}")
    if duplicate_normalized:
        errors.append("ladder prompt text leakage detected")
    return {
        "ok": not errors,
        "suite_version": _LADDER_SUITE_VERSION,
        "selection_count": len(split_sets["selection"]),
        "report_count": len(split_sets["report"]),
        "holdout_count": len(split_sets["holdout"]),
        "duplicate_normalized_text_across_splits": duplicate_normalized,
        "errors": errors,
    }


def _assert_prompt_registry_valid() -> None:
    all_defined_prompts = (
        _BASE_QUALITY_PROMPTS
        + _LANE0_WIDENED_EXTRA_PROMPTS
        + _AIR_VQ_LADDER_SELECT_V1_PROMPTS
        + _AIR_VQ_LADDER_REPORT_V1_PROMPTS
        + _AIR_VQ_LADDER_HOLDOUT_V1_PROMPTS
        + _AIR_IMATRIX_CALIB_V1_PROMPTS
    )
    all_ids = [prompt.prompt_id for prompt in all_defined_prompts]
    if len(all_ids) != len(set(all_ids)):
        seen: set[str] = set()
        duplicates: list[str] = []
        for prompt_id in all_ids:
            if prompt_id in seen:
                duplicates.append(prompt_id)
            else:
                seen.add(prompt_id)
        raise RuntimeError(f"duplicate quality prompt ids: {sorted(duplicates)}")
    for name, prompts in _PROMPT_SETS.items():
        if not prompts:
            raise RuntimeError(f"quality prompt set {name!r} is empty")
        for prompt in prompts:
            if not prompt.prompt_id or not prompt.text:
                raise RuntimeError(f"invalid quality prompt in set {name!r}: {prompt!r}")
            if prompt.max_new_tokens <= 0:
                raise RuntimeError(f"invalid max_new_tokens for {prompt.prompt_id!r}")
            if prompt.context_tokens is not None and prompt.context_tokens <= 0:
                raise RuntimeError(f"invalid context_tokens for {prompt.prompt_id!r}")
            if name.startswith("air_vq_ladder_"):
                if prompt.eval_split not in {"selection", "report", "holdout"}:
                    raise RuntimeError(f"invalid eval_split for {prompt.prompt_id!r}")
                if prompt.suite_version != _LADDER_SUITE_VERSION:
                    raise RuntimeError(f"invalid suite_version for {prompt.prompt_id!r}")
                if not prompt.row_intent:
                    raise RuntimeError(f"missing row_intent for {prompt.prompt_id!r}")
            if name == "air_imatrix_calib_v1":
                if prompt.eval_split is not None:
                    raise RuntimeError(f"imatrix prompt {prompt.prompt_id!r} must not have eval_split")
                if prompt.suite_version != _IMATRIX_SUITE_VERSION:
                    raise RuntimeError(f"invalid suite_version for {prompt.prompt_id!r}")
                if not prompt.row_intent or not prompt.row_intent.startswith("imatrix_"):
                    raise RuntimeError(f"invalid row_intent for {prompt.prompt_id!r}")
    split_report = validate_quality_prompt_splits()
    if not split_report["ok"]:
        raise RuntimeError(f"invalid ladder prompt splits: {split_report['errors']}")
    imatrix_report = validate_imatrix_calibration_prompt_set()
    if not imatrix_report["ok"]:
        raise RuntimeError(f"invalid imatrix calibration prompts: {imatrix_report['errors']}")


_assert_prompt_registry_valid()
