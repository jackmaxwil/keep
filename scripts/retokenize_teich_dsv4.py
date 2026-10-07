#!/usr/bin/env python3
"""Re-tokenize the teich coding-agent corpus from GLM-5.2 onto the DeepSeek-V4-Flash grid.

Wave 1 of the KEEP DeepSeek-V4-Flash compounding campaign
(``docs/deepseek-v4-flash/2026-08-11-campaign-plan.md``). Produces a re-tokenized
prompt pack (not committed, ~100 MB class) plus a committed 5-way split
manifest.

=============================================================================
DESIGN DECISIONS
=============================================================================

1. Supervision boundary convention -- SHIFT-BY-ONE (next-token targets)
----------------------------------------------------------------------
Verified empirically on all 257 rows and confirmed against the original
builder (``src/keep/quality/teich_corpus.py::build_keep_prompt_row``):

    target_token_ids[i] == encoded_token_ids[positions[i] + 1]     (257/257)
    target_token_ids[i] == encoded_token_ids[positions[i]]         (0/257)

The builder docstring states it outright: "Assistant-character spans identify
target tokens. KEEP positions index the logits that predict those tokens, so
each target token at index ``i`` maps to position ``i - 1`` and target
``input_ids[i]``."

So ``positions`` are *logit* indices, not token indices. The quantity that is
intrinsic to the text -- and therefore the quantity that must be preserved
across a change of token grid -- is the **supervised target set**

    T = { p + 1 : p in positions }

i.e. the indices of the tokens whose prediction is trained. This script
segments on membership in T, re-encodes, recomputes T' on the V4 grid, and
emits ``positions' = sorted(t - 1 for t in T')`` with
``target_token_ids' = [ids'[p + 1] for p in positions']``. That is a bijection
T' <-> positions', so the convention is preserved exactly.

Corollary used below: index 0 is never a supervised target
(min(positions) == 1728 over the corpus, so min(T) == 1729), which makes the
``t - 1`` step total.

Note that the original builder skipped zero-width tokens when computing target
indices (``if token_end > token_start``), so GLM tokens flagged
``special: true`` in ``tokenizer.json`` (the role markers ``[gMASK]``,
``<sop>``, ``<|system|>``, ``<|user|>``, ``<|assistant|>``,
``<|observation|>``) are never supervised targets. Non-``special`` added tokens
can be and are.

Measured over the 9,836 supervised runs in the corpus, supervision starts at:

    <tool_call>       5,215   53.0 %
    ordinary text     2,426   24.7 %
    <think>           2,195   22.3 %

so it is **not** true that supervision begins at ``<think>`` -- that holds for
under a quarter of runs, and for the ``claude`` provider it never happens
(59 % ordinary text, 41 % ``<tool_call>``). Per-provider run starts:

    claude  n=439    ordinary 59 %  <tool_call> 41 %  <think>  0 %
    codex   n=3937   <think>  35 %  <tool_call> 34 %  ordinary 32 %
    cursor  n=5460   <tool_call> 68 %  ordinary 17 %  <think> 15 %

``<tool_response>``/``</tool_response>`` are never supervised (0 of 11,953):
tool results are environment output, not model output.

2. Segment-wise decode: verified, not structurally guaranteed
-------------------------------------------------------------
Both tokenizers are ByteLevel BPE with no normalizer (V4's is an empty
``Sequence``) and a ByteLevel post-processor that inserts no BOS/EOS. V4's
pre-tokenizer ByteLevel has ``add_prefix_space: false``, so re-encoding a
segment does not inject a leading space. ``add_special_tokens=False`` is passed
on every V4 encode so no BOS is inserted mid-stream.

Because 24.7 % of supervised runs start on ordinary text, segment boundaries
*do* fall mid-text, and a boundary could in principle split a multi-byte UTF-8
character (yielding U+FFFD on each side) or split a word so that the two halves
re-encode differently than the whole would. Neither is ruled out by
construction. What rules it out here is measurement: the per-session roundtrip
gate below requires **exact string equality** (``--max-char-divergence``
defaults to ``0.0``), and all 257 sessions pass it. Any session where a
boundary did damage the text would be excluded loudly rather than shipped.

3. Tier C folding -- do not fence literal text into its own segment
-------------------------------------------------------------------
A GLM special that is re-encoded as literal text (Tier C below, and Tier B
under ``--tier-b-mode literal``) must **not** get its own segment. If it did,
the text on either side of it could never BPE-merge across it, producing a
token stream V4 would never emit for that text -- gratuitous off-distribution
drift.

So segment identity keys on ``(supervised_flag, is_native_special)`` where
``is_native_special`` is true only for mappings applied as a real V4 token id.
Literal-text mappings are treated as ordinary text, so they fold into their
neighbours and merge normally. Folding never crosses a supervision-flag change
(the flag is part of the key), so the supervised/unsupervised chunk sequence is
bit-identical to the unfolded segmentation -- it is strictly a merge of
adjacent same-flag segments.

Residual drift is measured, not assumed: ``non_canonical_token_overhead`` in
the manifest is
``(sum(len(stored_stream)) - sum(len(v4.encode(session_text)))) /
sum(len(v4.encode(session_text)))``, i.e. how much longer the stored streams
are than what V4's tokenizer would natively produce from the identical text.

**Measured result with folding: 0.0, and the streams are identical
element-wise, not merely equal in length** (``canonical_stream_sessions ==
sessions_included`` in the manifest, asserted per session). The Tier A/B
substitutions land on exactly the ids V4's own tokenizer emits for those
surface forms, because every V4 target is an entry in V4's ``added_tokens``.
So the stored stream *is* the canonical V4 encoding of its own text: there is
no off-distribution tokenization drift left to trade off. Without folding the
same measurement reads +0.695 %.

4. GLM -> V4 special-token mapping
----------------------------------
17 distinct GLM special/added ids occur in the corpus (census asserted at
runtime against ``GLM_SPECIAL_MAP``; an unmapped occurring id aborts the run
and is listed). Each is assigned to one of three tiers:

  Tier A -- V4's *live* chat convention as implemented in
    ``DeepSeek-V4-Flash-0731/encoding/encoding_dsv4.py``. Safest mapping:
    these ids are what a V4 chat prompt actually contains.

      [gMASK] + <sop>  ->  <|begin_of_sentence|>   (pair rule, see below)
      <|user|>         ->  <|User|>          128803
      <|assistant|>    ->  <|Assistant|>     128804
      <think>          ->  <think>           128821   (identical surface)
      </think>         ->  </think>          128822   (identical surface)

  Tier B -- V4's vocabulary carries a dedicated token for the same structural
    role, but ``encoding_dsv4.py`` does not exercise it (V4 expresses tool
    calls in DSML text instead, and puts system content directly after BOS
    with no marker). These are DeepSeek-V3-lineage tokens that remain in the
    V4 vocab.

      <|system|>        ->  <|begin_sys|>            128826
      <|observation|>   ->  <|tool_outputs_begin|>   128810
      <tool_call>       ->  <|tool_call_begin|>      128808
      </tool_call>      ->  <|tool_call_end|>        128809
      <tool_response>   ->  <|tool_output_begin|>    128812
      </tool_response>  ->  <|tool_output_end|>      128813
      <|image|>         ->  <|image|>                129279

    Because "dormant vocab entry" vs. "trained marker" is not decidable from
    the artifacts on hand, ``--tier-b-mode literal`` flips every Tier B entry
    to the Tier C treatment without a code edit, so Wave 2 can A/B it.

  Tier C -- no V4 counterpart exists. The surface string is preserved
    byte-exactly and re-encoded as ordinary V4 BPE text (a handful of tokens
    each). This is explicitly *not* a drop: the bytes survive; only their
    tokenization changes.

      <arg_key> </arg_key> <arg_value> </arg_value>

    GLM's tool-call argument encoding is positional
    (``<arg_key>k</arg_key><arg_value>v</arg_value>``) whereas V4's DSML is
    attribute-based (``<|DSML|>parameter name="k" string="true">v<...``).
    Converting between them *moves text across the supervision boundary* and
    is a semantic re-serialization, not a re-tokenization -- deliberately out
    of scope here and flagged for Wave 2. These four tags are each supervised
    40,239 times (of 40,753 occurrences), so before Tier C folding they alone
    contributed 4 x 40,239 x 3 = 482,868 excess supervised tokens.

  BOS pair rule: GLM opens every session with the two-token idiom
  ``[gMASK] <sop>``; V4 has a single BOS. Verified invariant over all 257
  rows: id 154822 occurs only at index 0 and id 154824 only at index 1, and
  neither is ever a supervised target. The ordered pair therefore collapses to
  the single token ``<|begin_of_sentence|>`` (id 0) as one enumerated rule --
  not a silent per-token drop.

5. Roundtrip verification
-------------------------
Per session: build the expected V4 text by taking the GLM decode of the full
stream and substituting each mapped GLM surface form with its V4 surface form
(identity for literal-text mappings), then require it to equal
``v4.decode(new_ids, skip_special_tokens=False)``. Character divergence is
measured as the fraction of the string outside the common prefix/suffix (a
cheap conservative upper bound on the edit region -- exact edit distance on
100-300 KB strings x 257 sessions is not worth the runtime). Any session over
``--max-char-divergence`` (**default 0.0**, i.e. exact equality) is excluded
loudly with its divergence recorded in the manifest.

6. Splits
---------
The campaign's 5-way assignment lands in ``campaign_split``. Sessions are
grouped by ``provenance.source_session_id`` so a session can never span two
splits (all 257 ids are distinct in this corpus, so every group has size 1,
but the grouping is implemented for correctness under future re-windowing).
Groups are ordered by ``sha256(SPLIT_SEED | representative_prompt_id)`` --
with size-1 groups that is exactly a deterministic hash of ``prompt_id`` --
then filled against the quotas below, scaled proportionally (largest-remainder,
fixed split order for ties) if fewer than 257 sessions survive; a scaling that
would empty any split aborts the run.

    calibration 40 | mtp-train 120 | report 30 | selection 30 | holdout 37

7. ``split`` / ``tuning_eligible`` are RECOMPUTED, not inherited
---------------------------------------------------------------
The GLM-era pack carried ``split`` in {"train","holdout"} and
``tuning_eligible == (split == "train")``. Live consumers gate on exactly those
two field names:

    src/keep/quality/glm52_adapter_training.py:506   teich rows: requires
        split == "train" and tuning_eligible is True to permit tuning
    src/keep/quality/glm52_teich_training_cache.py:507   split vocabulary is
        {"train","validation","holdout"}, tuning_eligible == (split == "train")
    src/keep/quality/glm52_recovery.py:160          requires split ==
        "selection" and tuning_eligible is True
    src/keep/quality/glm52_teacher_cache.py:560     requires
        tuning_eligible == (split == "selection")

Carrying the GLM-era values through verbatim would leave 35 of the 37 new
holdout sessions labelled ``split="train", tuning_eligible=True``, so any
consumer that had not yet learned about ``campaign_split`` would train on the
V4 holdout. That is a leakage landmine, so both fields are **derived from
``campaign_split``**:

    tuning_eligible = campaign_split in {"calibration", "mtp-train"}
    split           = "train" if tuning_eligible else "holdout"

``"holdout"`` (not a novel value like ``"eval"``) is deliberate: it is inside
the teich consumer vocabulary, so a stale reader takes the *safe* branch rather
than hitting an unknown label. Checked against each gate above, a stale reader
now either refuses the row or raises -- never silently trains on it.

The GLM-era values survive as ``glm_era_split`` and
``glm_era_tuning_eligible`` so provenance is not lost, under names no consumer
gates on.

8. token_ids_sha256
-------------------
Recomputed per row with the corpus's own recipe, reused from
``keep.quality.teich_corpus`` to keep one definition:
``sha256(json.dumps(ids, sort_keys=True, separators=(",", ":"),
ensure_ascii=False))``. Verified to reproduce all 257 source digests.

Determinism: no randomness, no dict-order dependence, no wall-clock in any
hashed payload. Re-running on the same inputs reproduces both output files
byte-for-byte.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from keep.quality.teich_corpus import _token_ids_sha256  # noqa: E402

SCRIPT_VERSION = "retokenize_teich_dsv4/2.0.0"

DEFAULT_SOURCE_PACK = Path(
    "/Users/jack.mazac/models/teich/glm52-coding-agent-initial-v2-20260713.json"
)
DEFAULT_OUTPUT_PACK = Path("/Users/jack.mazac/models/teich/dsv4-coding-agent-v1-20260811.json")
DEFAULT_MANIFEST = REPO_ROOT / "recipes" / "dsv4_teich_split_manifest_v1_20260811.json"
DEFAULT_V4_TOKENIZER_DIR = Path("/Users/jack.mazac/models/DeepSeek-V4-Flash-0731")

DEFAULT_GLM_TOKENIZER_REPO = "0xSero/glm-5.2-reap-504B-v2"
GLM_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")

# Expected identities, all hard asserts so a swapped tokenizer cannot slip
# through. The GLM *model* vocab figure is the padded embedding width
# (config.json vocab_size), larger than the tokenizer's real token count.
V4_TOKENIZER_SHA256 = "8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf"
V4_VOCAB_SIZE = 129280
GLM_TOKENIZER_SHA256 = "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d"
GLM_MODEL_VOCAB_SIZE = 154880
GLM_TOKENIZER_VOCAB_SIZE = 154856

# campaign_split values whose rows may be trained/fitted on. Everything else is
# eval-only and must never be tuning_eligible.
TRAIN_CAMPAIGN_SPLITS = frozenset({"calibration", "mtp-train"})
EVAL_SPLIT_LABEL = "holdout"
TRAIN_SPLIT_LABEL = "train"

V4_MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash-0731"

SPLIT_SEED = "keep-v4flash-wave1-teich-split-v1"
SPLIT_QUOTAS: tuple[tuple[str, int], ...] = (
    ("calibration", 40),
    ("mtp-train", 120),
    ("report", 30),
    ("selection", 30),
    ("holdout", 37),
)

GLM_BOS_PAIR = (154822, 154824)  # [gMASK] <sop>

TIER_A = "A:v4-live-convention"
TIER_B = "B:v4-vocab-same-role"
TIER_C = "C:no-v4-counterpart-literal-text"

# glm_id -> (glm surface, v4 surface or None for literal, tier)
GLM_SPECIAL_MAP: dict[int, tuple[str, str | None, str]] = {
    154826: ("<|system|>", "<｜begin▁sys｜>", TIER_B),
    154827: ("<|user|>", "<｜User｜>", TIER_A),
    154828: ("<|assistant|>", "<｜Assistant｜>", TIER_A),
    154829: ("<|observation|>", "<｜tool▁outputs▁begin｜>", TIER_B),
    154841: ("<think>", "<think>", TIER_A),
    154842: ("</think>", "</think>", TIER_A),
    154843: ("<tool_call>", "<｜tool▁call▁begin｜>", TIER_B),
    154844: ("</tool_call>", "<｜tool▁call▁end｜>", TIER_B),
    154845: ("<tool_response>", "<｜tool▁output▁begin｜>", TIER_B),
    154846: ("</tool_response>", "<｜tool▁output▁end｜>", TIER_B),
    154847: ("<arg_key>", None, TIER_C),
    154848: ("</arg_key>", None, TIER_C),
    154849: ("<arg_value>", None, TIER_C),
    154850: ("</arg_value>", None, TIER_C),
    154854: ("<|image|>", "<｜image｜>", TIER_B),
}
V4_BOS_SURFACE = "<｜begin▁of▁sentence｜>"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_glm_tokenizer(cache_dir: Path, repo: str) -> tuple[Any, dict[str, str], str]:
    """Fetch *only* GLM's tokenizer files (never a full snapshot) and load them.

    One repo, no silent fallback: a different repo must be named explicitly via
    ``--glm-tokenizer-repo``, and its tokenizer.json still has to match
    ``GLM_TOKENIZER_SHA256``.
    """
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        name: Path(hf_hub_download(repo, name, local_dir=str(cache_dir)))
        for name in GLM_TOKENIZER_FILES
    }
    digest = sha256_file(paths["tokenizer.json"])
    if digest != GLM_TOKENIZER_SHA256:
        raise SystemExit(
            f"GLM tokenizer.json sha256 mismatch for {repo}:\n"
            f"  got      {digest}\n  expected {GLM_TOKENIZER_SHA256}"
        )
    tokenizer = Tokenizer.from_file(str(paths["tokenizer.json"]))
    vocab = tokenizer.get_vocab_size(True)
    if vocab != GLM_TOKENIZER_VOCAB_SIZE:
        raise SystemExit(
            f"GLM tokenizer from {repo} has vocab {vocab}, expected "
            f"{GLM_TOKENIZER_VOCAB_SIZE} (model vocab_size {GLM_MODEL_VOCAB_SIZE})"
        )
    return tokenizer, {name: sha256_file(path) for name, path in paths.items()}, repo


def load_v4_tokenizer(tokenizer_dir: Path) -> tuple[Any, dict[str, str]]:
    from tokenizers import Tokenizer

    tokenizer_json = tokenizer_dir / "tokenizer.json"
    config_json = tokenizer_dir / "tokenizer_config.json"
    digest = sha256_file(tokenizer_json)
    if digest != V4_TOKENIZER_SHA256:
        raise SystemExit(
            f"V4 tokenizer.json sha256 mismatch:\n  got      {digest}\n  expected {V4_TOKENIZER_SHA256}"
        )
    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    vocab = tokenizer.get_vocab_size(True)
    if vocab != V4_VOCAB_SIZE:
        raise SystemExit(f"V4 tokenizer vocab {vocab}, expected {V4_VOCAB_SIZE}")
    return tokenizer, {
        "tokenizer.json": digest,
        "tokenizer_config.json": sha256_file(config_json),
    }


def build_special_plan(v4: Any, tier_b_mode: str) -> dict[int, dict[str, Any]]:
    """Resolve the mapping table into concrete V4 id sequences."""
    plan: dict[int, dict[str, Any]] = {}
    for glm_id, (glm_surface, v4_surface, tier) in sorted(GLM_SPECIAL_MAP.items()):
        as_literal = v4_surface is None or (tier == TIER_B and tier_b_mode == "literal")
        if as_literal:
            ids = v4.encode(glm_surface, add_special_tokens=False).ids
            if not ids:
                raise SystemExit(f"literal mapping for {glm_surface!r} encoded to nothing")
            plan[glm_id] = {
                "glm_surface": glm_surface,
                "v4_surface": glm_surface,
                "v4_ids": ids,
                "tier": tier,
                "applied_as": "literal_text",
            }
            continue
        token_id = v4.token_to_id(v4_surface)
        if token_id is None:
            raise SystemExit(
                f"V4 tokenizer has no token {v4_surface!r} for GLM {glm_surface!r} "
                f"(id {glm_id}); mapping table is stale"
            )
        plan[glm_id] = {
            "glm_surface": glm_surface,
            "v4_surface": v4_surface,
            "v4_ids": [token_id],
            "tier": tier,
            "applied_as": "v4_special_token",
        }
    return plan


def char_divergence(expected: str, actual: str) -> float:
    """Fraction of the longer string outside the shared prefix/suffix."""
    if expected == actual:
        return 0.0
    longest = max(len(expected), len(actual))
    if longest == 0:
        return 0.0
    limit = min(len(expected), len(actual))
    prefix = 0
    while prefix < limit and expected[prefix] == actual[prefix]:
        prefix += 1
    suffix = 0
    while suffix < limit - prefix and expected[-1 - suffix] == actual[-1 - suffix]:
        suffix += 1
    return (longest - prefix - suffix) / longest


def retokenize_row(
    row: dict[str, Any],
    *,
    glm: Any,
    v4: Any,
    plan: dict[int, dict[str, Any]],
    max_char_divergence: float,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Return (retokenized row or None, diagnostics)."""
    prompt_id = row["prompt_id"]
    glm_ids: list[int] = row["encoded_token_ids"]
    glm_positions: list[int] = row["positions"]

    if len(glm_ids) != row["token_count"]:
        return None, {"prompt_id": prompt_id, "reason": "token_count disagrees with stream length"}
    if _token_ids_sha256(glm_ids) != row["token_ids_sha256"]:
        return None, {"prompt_id": prompt_id, "reason": "source token_ids_sha256 mismatch"}
    if tuple(glm_ids[:2]) != GLM_BOS_PAIR:
        return None, {"prompt_id": prompt_id, "reason": "stream does not open with [gMASK]<sop>"}
    if 154822 in glm_ids[1:] or 154824 in glm_ids[2:]:
        return None, {"prompt_id": prompt_id, "reason": "BOS-pair ids recur mid-stream"}
    if len(glm_positions) != len(row["target_token_ids"]):
        return None, {"prompt_id": prompt_id, "reason": "positions/target length mismatch"}
    if not glm_positions:
        return None, {"prompt_id": prompt_id, "reason": "row has no supervised positions"}
    # Range-check before indexing glm_ids[p + 1] so a malformed row is a clean
    # exclusion rather than an IndexError that kills the whole run.
    if any(p < 0 or p + 1 >= len(glm_ids) for p in glm_positions):
        return None, {"prompt_id": prompt_id, "reason": "supervised position out of range"}
    if any(glm_ids[p + 1] != t for p, t in zip(glm_positions, row["target_token_ids"])):
        return None, {"prompt_id": prompt_id, "reason": "shift-by-one supervision invariant broken"}

    supervised = set(p + 1 for p in glm_positions)
    if 0 in supervised or 1 in supervised:
        return None, {"prompt_id": prompt_id, "reason": "BOS pair is marked supervised"}

    # Maximal runs of constant (supervised flag, native-special-vs-text), after
    # the BOS pair. Only mappings applied as a real V4 token id fence a segment;
    # literal-text mappings count as text so they fold into their neighbours and
    # BPE-merge normally (see design note 3). The supervision flag is part of the
    # key, so folding never crosses a supervision boundary.
    native_special = {glm_id for glm_id, e in plan.items() if e["applied_as"] == "v4_special_token"}
    segments: list[tuple[bool, bool, list[int]]] = []
    for index in range(2, len(glm_ids)):
        token = glm_ids[index]
        key = (index in supervised, token in native_special)
        if segments and (segments[-1][0], segments[-1][1]) == key:
            segments[-1][2].append(token)
        else:
            segments.append((key[0], key[1], [token]))

    new_ids: list[int] = [0]  # <|begin_of_sentence|> replaces [gMASK]<sop>
    new_supervised: list[bool] = [False]
    for is_supervised, is_special, tokens in segments:
        if is_special:
            emitted: list[int] = []
            for token in tokens:
                emitted.extend(plan[token]["v4_ids"])
        else:
            text = glm.decode(tokens, skip_special_tokens=False)
            emitted = v4.encode(text, add_special_tokens=False).ids
        if not emitted:
            return None, {"prompt_id": prompt_id, "reason": "segment re-encoded to zero tokens"}
        new_ids.extend(emitted)
        new_supervised.extend([is_supervised] * len(emitted))

    new_targets = [index for index, flag in enumerate(new_supervised) if flag]
    if not new_targets or new_targets[0] == 0:
        return None, {"prompt_id": prompt_id, "reason": "no usable supervised targets on V4 grid"}
    new_positions = [index - 1 for index in new_targets]
    new_target_ids = [new_ids[p + 1] for p in new_positions]

    # Roundtrip: GLM text with mapped surfaces substituted must equal the V4 decode.
    glm_text = glm.decode(glm_ids, skip_special_tokens=False)
    expected = glm_text.replace(
        plan_bos_surface(glm), V4_BOS_SURFACE, 1
    )
    for glm_id in sorted(plan, key=lambda i: -len(plan[i]["glm_surface"])):
        entry = plan[glm_id]
        if entry["v4_surface"] != entry["glm_surface"]:
            expected = expected.replace(entry["glm_surface"], entry["v4_surface"])
    actual = v4.decode(new_ids, skip_special_tokens=False)
    divergence = char_divergence(expected, actual)
    if divergence > max_char_divergence:
        return None, {
            "prompt_id": prompt_id,
            "reason": "roundtrip character divergence above threshold",
            "char_divergence": divergence,
            "expected_chars": len(expected),
            "actual_chars": len(actual),
        }

    # What V4's tokenizer would natively emit for this exact text. Any gap is
    # off-distribution tokenization drift introduced by the explicit special
    # mapping; with literal-text folding the streams come out identical.
    canonical_ids = v4.encode(expected, add_special_tokens=False).ids
    canonical_tokens = len(canonical_ids)
    is_canonical = canonical_ids == new_ids

    new_row = dict(row)
    # Retire the GLM-era gate fields under names nothing gates on; the live
    # values are recomputed from campaign_split once splits are assigned.
    new_row["glm_era_split"] = row["split"]
    new_row["glm_era_tuning_eligible"] = row["tuning_eligible"]
    new_row.pop("split", None)
    new_row.pop("tuning_eligible", None)
    new_row["encoded_token_ids"] = new_ids
    new_row["positions"] = new_positions
    new_row["target_token_ids"] = new_target_ids
    new_row["token_count"] = len(new_ids)
    new_row["token_ids_sha256"] = _token_ids_sha256(new_ids)
    return new_row, {
        "prompt_id": prompt_id,
        "char_divergence": divergence,
        "glm_token_count": len(glm_ids),
        "glm_supervised": len(glm_positions),
        "v4_token_count": len(new_ids),
        "v4_supervised": len(new_positions),
        "v4_canonical_token_count": canonical_tokens,
        "v4_stream_is_canonical": is_canonical,
    }


def plan_bos_surface(glm: Any) -> str:
    return glm.decode(list(GLM_BOS_PAIR), skip_special_tokens=False)


def scaled_quotas(total: int) -> dict[str, int]:
    """Largest-remainder scaling of SPLIT_QUOTAS to ``total`` sessions."""
    declared = sum(count for _, count in SPLIT_QUOTAS)
    if total == declared:
        return dict(SPLIT_QUOTAS)
    exact = [(name, total * count / declared) for name, count in SPLIT_QUOTAS]
    quotas = {name: int(value) for name, value in exact}
    remaining = total - sum(quotas.values())
    order = sorted(
        range(len(exact)),
        key=lambda i: (-(exact[i][1] - int(exact[i][1])), i),
    )
    for i in order[:remaining]:
        quotas[exact[i][0]] += 1
    empty = sorted(name for name, count in quotas.items() if count < 1)
    if empty:
        raise SystemExit(
            f"only {total} sessions survived, which starves split(s) {empty} to zero; "
            f"the campaign's eval contract needs every split non-empty -- "
            f"fix the exclusions or re-cut the quotas deliberately"
        )
    return quotas


def assign_campaign_splits(rows: list[dict[str, Any]]) -> dict[str, str]:
    """prompt_id -> campaign_split, session-disjoint and deterministic."""
    groups: dict[str, list[str]] = {}
    for row in rows:
        groups.setdefault(row["provenance"]["source_session_id"], []).append(row["prompt_id"])
    ordered = sorted(groups.items(), key=lambda item: sorted(item[1])[0])
    ordered.sort(
        key=lambda item: hashlib.sha256(
            f"{SPLIT_SEED}|{sorted(item[1])[0]}".encode()
        ).hexdigest()
    )
    quotas = scaled_quotas(len(ordered))
    assignment: dict[str, str] = {}
    cursor = 0
    for name, _ in SPLIT_QUOTAS:
        for _ in range(quotas[name]):
            if cursor >= len(ordered):
                break
            for prompt_id in ordered[cursor][1]:
                assignment[prompt_id] = name
            cursor += 1
    for group_key, prompt_ids in ordered[cursor:]:  # pragma: no cover - quota covers all
        for prompt_id in prompt_ids:
            assignment[prompt_id] = SPLIT_QUOTAS[-1][0]
        del group_key
    return assignment


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"), sort_keys=False)
        handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source-pack", type=Path, default=DEFAULT_SOURCE_PACK)
    parser.add_argument("--output-pack", type=Path, default=DEFAULT_OUTPUT_PACK)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--v4-tokenizer-dir", type=Path, default=DEFAULT_V4_TOKENIZER_DIR)
    parser.add_argument(
        "--glm-tokenizer-cache",
        type=Path,
        default=Path("/tmp/keep-glm52-tokenizer"),
        help="directory for the two downloaded GLM tokenizer files",
    )
    parser.add_argument(
        "--glm-tokenizer-repo",
        default=DEFAULT_GLM_TOKENIZER_REPO,
        help=(
            "HF repo to pull tokenizer.json/tokenizer_config.json from. No silent "
            "fallback: whatever is named here must still match GLM_TOKENIZER_SHA256."
        ),
    )
    parser.add_argument(
        "--max-char-divergence",
        type=float,
        default=0.0,
        help="per-session roundtrip tolerance; 0.0 means exact string equality (default)",
    )
    parser.add_argument("--tier-b-mode", choices=("native", "literal"), default="native")
    args = parser.parse_args(argv)

    glm, glm_digests, glm_repo = load_glm_tokenizer(
        args.glm_tokenizer_cache, args.glm_tokenizer_repo
    )
    v4, v4_digests = load_v4_tokenizer(args.v4_tokenizer_dir)
    plan = build_special_plan(v4, args.tier_b_mode)
    print(f"GLM tokenizer: {glm_repo} vocab {glm.get_vocab_size(True)}")
    print(f"V4  tokenizer: {args.v4_tokenizer_dir} vocab {v4.get_vocab_size(True)}")

    source = json.loads(args.source_pack.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = source["prompt_rows"]
    print(f"loaded {len(rows)} rows from {args.source_pack}")

    # Census: every GLM special/added id occurring in the corpus must be mapped.
    census: Counter[int] = Counter()
    for row in rows:
        for token in row["encoded_token_ids"]:
            if token >= 154820:
                census[token] += 1
    known = set(plan) | set(GLM_BOS_PAIR)
    unmapped = sorted(set(census) - known)
    if unmapped:
        detail = ", ".join(
            f"{tid} {glm.decode([tid], skip_special_tokens=False)!r} x{census[tid]}"
            for tid in unmapped
        )
        raise SystemExit(f"unmapped GLM special tokens occur in the corpus: {detail}")
    print(f"special-token census: {len(census)} distinct ids, all mapped")

    kept: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        new_row, info = retokenize_row(
            row, glm=glm, v4=v4, plan=plan, max_char_divergence=args.max_char_divergence
        )
        if new_row is None:
            exclusions.append(info)
            print(f"  [{index}/{len(rows)}] EXCLUDE {info['prompt_id']}: {info['reason']}")
            continue
        kept.append(new_row)
        diagnostics.append(info)
        if index % 25 == 0 or index == len(rows):
            print(f"  [{index}/{len(rows)}] retokenized")

    if not kept:
        raise SystemExit("every session was excluded; refusing to write empty outputs")

    assignment = assign_campaign_splits(kept)
    for row in kept:
        campaign_split = assignment[row["prompt_id"]]
        eligible = campaign_split in TRAIN_CAMPAIGN_SPLITS
        row["campaign_split"] = campaign_split
        # Recomputed, never inherited -- see design note 7. Carrying the GLM-era
        # values here would label 35 of 37 holdout rows train/eligible.
        row["split"] = TRAIN_SPLIT_LABEL if eligible else EVAL_SPLIT_LABEL
        row["tuning_eligible"] = eligible

    leaked = [
        row["prompt_id"]
        for row in kept
        if row["tuning_eligible"] and row["campaign_split"] not in TRAIN_CAMPAIGN_SPLITS
    ]
    if leaked:
        raise SystemExit(f"eval-split rows marked tuning_eligible: {leaked[:5]}")

    glm_lengths = sorted(info["glm_token_count"] for info in diagnostics)
    v4_lengths = sorted(info["v4_token_count"] for info in diagnostics)
    canonical_total = sum(info["v4_canonical_token_count"] for info in diagnostics)
    totals = {
        "sessions_in": len(rows),
        "sessions_included": len(kept),
        "sessions_excluded": len(exclusions),
        "glm_raw_tokens": sum(glm_lengths),
        "glm_supervised_tokens": sum(info["glm_supervised"] for info in diagnostics),
        "v4_raw_tokens": sum(v4_lengths),
        "v4_supervised_tokens": sum(info["v4_supervised"] for info in diagnostics),
        "glm_session_tokens_min": glm_lengths[0],
        "glm_session_tokens_median": int(statistics.median(glm_lengths)),
        "glm_session_tokens_max": glm_lengths[-1],
        "v4_session_tokens_min": v4_lengths[0],
        "v4_session_tokens_median": int(statistics.median(v4_lengths)),
        "v4_session_tokens_max": v4_lengths[-1],
        "max_char_divergence_observed": max(info["char_divergence"] for info in diagnostics),
        "v4_canonical_raw_tokens": canonical_total,
        "canonical_stream_sessions": sum(
            1 for info in diagnostics if info["v4_stream_is_canonical"]
        ),
        "sessions_over_65536_v4": sum(1 for length in v4_lengths if length > 65536),
        "sessions_over_65536_glm": sum(1 for length in glm_lengths if length > 65536),
    }
    totals["v4_raw_ratio"] = totals["v4_raw_tokens"] / totals["glm_raw_tokens"]
    totals["v4_supervised_ratio"] = (
        totals["v4_supervised_tokens"] / totals["glm_supervised_tokens"]
    )
    # How much longer the stored streams are than a native V4 encoding of the
    # identical text. Cannot be 0 while any mapping substitutes a different
    # surface form; measured rather than assumed.
    totals["non_canonical_token_overhead"] = (
        totals["v4_raw_tokens"] - canonical_total
    ) / canonical_total
    totals["canonical_vs_glm_ratio"] = canonical_total / totals["glm_raw_tokens"]

    retokenization = {
        "script_version": SCRIPT_VERSION,
        "source_pack": str(args.source_pack),
        "source_pack_sha256": sha256_file(args.source_pack),
        "source_model_id": source["model_id"],
        "supervision_convention": "positions[i] indexes the logit predicting encoded_token_ids[positions[i] + 1]",
        "tier_b_mode": args.tier_b_mode,
        "max_char_divergence": args.max_char_divergence,
        "split_seed": SPLIT_SEED,
        "literal_text_folding": (
            "GLM specials re-encoded as literal text are treated as ordinary text when "
            "segmenting, so they fold into neighbouring same-supervision-flag segments and "
            "BPE-merge normally instead of being fenced into their own segment. Folding "
            "never crosses a supervision-flag change, so the supervised/unsupervised chunk "
            "sequence is identical to the unfolded segmentation."
        ),
        "split_field_policy": {
            "rule": (
                "split and tuning_eligible are RECOMPUTED from campaign_split, never "
                "inherited from the GLM-era pack."
            ),
            "tuning_eligible_rule": "campaign_split in train_campaign_splits",
            "train_campaign_splits": sorted(TRAIN_CAMPAIGN_SPLITS),
            "train_split_label": TRAIN_SPLIT_LABEL,
            "eval_split_label": EVAL_SPLIT_LABEL,
            "eval_split_label_rationale": (
                "'holdout' is inside the teich consumer split vocabulary "
                "{train, validation, holdout}, so a consumer that has not learned about "
                "campaign_split takes the safe branch instead of hitting an unknown label."
            ),
            "glm_era_fields_preserved": ["glm_era_split", "glm_era_tuning_eligible"],
            "consumer_gates_checked": [
                "src/keep/quality/glm52_adapter_training.py:506",
                "src/keep/quality/glm52_recovery.py:160",
                "src/keep/quality/glm52_teich_training_cache.py:507",
                "src/keep/quality/glm52_teacher_cache.py:560",
            ],
            "leak_averted": (
                "Inheriting the GLM-era values would have labelled 35 of 37 campaign-holdout "
                "sessions split='train', tuning_eligible=true."
            ),
        },
        "glm_tokenizer": {
            "repo": glm_repo,
            "vocab_size": glm.get_vocab_size(True),
            "model_vocab_size": GLM_MODEL_VOCAB_SIZE,
            "sha256": glm_digests,
        },
        "v4_tokenizer": {
            "path": str(args.v4_tokenizer_dir),
            "vocab_size": v4.get_vocab_size(True),
            "sha256": v4_digests,
        },
        "bos_pair_rule": {
            "glm_ids": list(GLM_BOS_PAIR),
            "glm_surface": plan_bos_surface(glm),
            "v4_ids": [0],
            "v4_surface": V4_BOS_SURFACE,
            "tier": TIER_A,
        },
        "special_token_map": {
            str(glm_id): {
                "glm_surface": entry["glm_surface"],
                "v4_surface": entry["v4_surface"],
                "v4_ids": entry["v4_ids"],
                "tier": entry["tier"],
                "applied_as": entry["applied_as"],
                "occurrences": census[glm_id],
            }
            for glm_id, entry in sorted(plan.items())
        },
    }

    pack = {
        "record_type": "dsv4_coding_agent_corpus",
        "model_id": V4_MODEL_ID,
        "created_date": "2026-08-11",
        "selection": source["selection"],
        "prompt_row_count": len(kept),
        "supervised_tokens": totals["v4_supervised_tokens"],
        "retokenization": retokenization,
        "prompt_rows": kept,
    }
    write_json(args.output_pack, pack)
    pack_sha = sha256_file(args.output_pack)
    print(f"wrote pack {args.output_pack} ({args.output_pack.stat().st_size} bytes) sha256 {pack_sha}")

    by_split: dict[str, list[dict[str, Any]]] = {name: [] for name, _ in SPLIT_QUOTAS}
    info_by_id = {info["prompt_id"]: info for info in diagnostics}
    for row in kept:
        by_split[row["campaign_split"]].append(row)
    splits_block: dict[str, Any] = {}
    for name, _ in SPLIT_QUOTAS:
        members = sorted(by_split[name], key=lambda r: r["prompt_id"])
        splits_block[name] = {
            "target_sessions": dict(SPLIT_QUOTAS)[name],
            "sessions": len(members),
            "v4_raw_tokens": sum(r["token_count"] for r in members),
            "v4_supervised_tokens": sum(len(r["positions"]) for r in members),
            "glm_raw_tokens": sum(info_by_id[r["prompt_id"]]["glm_token_count"] for r in members),
            "glm_supervised_tokens": sum(
                info_by_id[r["prompt_id"]]["glm_supervised"] for r in members
            ),
            "sessions_over_65536_v4": sum(1 for r in members if r["token_count"] > 65536),
            "split_label": TRAIN_SPLIT_LABEL
            if name in TRAIN_CAMPAIGN_SPLITS
            else EVAL_SPLIT_LABEL,
            "tuning_eligible": name in TRAIN_CAMPAIGN_SPLITS,
            "prompt_ids": [r["prompt_id"] for r in members],
        }

    manifest = {
        "record_type": "dsv4_teich_split_manifest",
        "manifest_version": "v2",
        "created_date": "2026-08-11",
        "campaign": "keep-v4flash-compounding-campaign-20260811",
        "wave": 1,
        "pack_file": str(args.output_pack),
        "pack_file_sha256": pack_sha,
        "pack_model_id": V4_MODEL_ID,
        "retokenization": retokenization,
        "totals": totals,
        "splits": splits_block,
        "exclusions": sorted(exclusions, key=lambda e: e["prompt_id"]),
        "rows": [
            {
                "prompt_id": row["prompt_id"],
                "campaign_split": row["campaign_split"],
                "split": row["split"],
                "tuning_eligible": row["tuning_eligible"],
                "glm_era_split": row["glm_era_split"],
                "glm_era_tuning_eligible": row["glm_era_tuning_eligible"],
                "source_session_id": row["provenance"]["source_session_id"],
                "provider": row["provenance"]["provider"],
                "topic_cluster": row["topic_cluster"],
                "token_ids_sha256": row["token_ids_sha256"],
                "v4_token_count": row["token_count"],
                "v4_supervised_tokens": len(row["positions"]),
                "v4_canonical_token_count": info_by_id[row["prompt_id"]]["v4_canonical_token_count"],
                "glm_token_count": info_by_id[row["prompt_id"]]["glm_token_count"],
                "glm_supervised_tokens": info_by_id[row["prompt_id"]]["glm_supervised"],
                "roundtrip_char_divergence": info_by_id[row["prompt_id"]]["char_divergence"],
            }
            for row in sorted(kept, key=lambda r: r["prompt_id"])
        ],
    }
    write_json(args.manifest, manifest)
    print(f"wrote manifest {args.manifest}")

    print("\n=== V4 vs GLM token statistics ===")
    print(f"sessions        {totals['sessions_included']} kept / {totals['sessions_excluded']} excluded")
    print(
        f"raw tokens      GLM {totals['glm_raw_tokens']:,} -> V4 {totals['v4_raw_tokens']:,} "
        f"({totals['v4_raw_ratio']:.4f}x)"
    )
    print(
        f"supervised      GLM {totals['glm_supervised_tokens']:,} -> V4 {totals['v4_supervised_tokens']:,} "
        f"({totals['v4_supervised_ratio']:.4f}x)"
    )
    print(
        f"session len     GLM {totals['glm_session_tokens_min']}/{totals['glm_session_tokens_median']}"
        f"/{totals['glm_session_tokens_max']} -> V4 {totals['v4_session_tokens_min']}"
        f"/{totals['v4_session_tokens_median']}/{totals['v4_session_tokens_max']} (min/median/max)"
    )
    print(f"max roundtrip char divergence  {totals['max_char_divergence_observed']:.3e}")
    print(
        f"canonical V4 encode of same text  {canonical_total:,} -> "
        f"non_canonical_token_overhead {totals['non_canonical_token_overhead']:+.4%}"
        f"  (element-wise canonical: {totals['canonical_stream_sessions']}"
        f"/{totals['sessions_included']} sessions)"
    )
    print(
        f"sessions over 65,536 tokens  GLM {totals['sessions_over_65536_glm']} -> "
        f"V4 {totals['sessions_over_65536_v4']}"
    )
    for name, _ in SPLIT_QUOTAS:
        block = splits_block[name]
        print(
            f"  {name:12s} {block['sessions']:3d} sessions  "
            f"{block['v4_raw_tokens']:,} raw  {block['v4_supervised_tokens']:,} supervised  "
            f"split={block['split_label']} eligible={block['tuning_eligible']} "
            f">64k={block['sessions_over_65536_v4']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
