# Wave 2, increment 3 — the DSpark drafter forward

**Status: complete and green.** The drafter runs. `mtp-train` (120 sessions,
5.55 M tokens, 47% of the corpus) is no longer blocked on a missing forward,
and the Wave-5 MTP verify runtime has both of its prerequisites: a drafter that
computes, and a `PoolingCache` that can honestly roll a rejected draft back.

Three things landed that binding and shapes could not express, and one that
increment 1 deliberately refused to fake:

1. **fp32 SwiGLU on the drafter only** — and *enforced*, not merely set.
2. **A non-causal `DSparkAttention`** that replaces `LocalAttention.__call__`
   rather than wrapping it.
3. **`main_proj` fed by an opt-in backbone tap** at `dspark_target_layer_ids`,
   with the default forward's return shape and cost untouched.
4. **`PoolingCache.is_trimmable()` is no longer a blanket `False`.** The
   one-update undo log is implemented and tested against exact state restore.

## Commits (branch `worktree-agent-ad6ea6fb6319343f1`)

| SHA | Subject |
| --- | --- |
| `304ada22` | `feat(ramp): DSpark drafter forward and the PoolingCache undo log` |
| `4de24988` | `feat(dsv4): wire --mode mtp-targets on the teacher runner` |

Files: `src/ramp/models/deepseek_v4_flash_adapter.py`,
`tests/test_deepseek_v4_flash_adapter.py`,
`src/keep/quality/dsv4_teacher_runner.py`,
`tests/test_dsv4_teacher_runner.py`,
`benchmarks/produce_dsv4_teacher_cache.py`.

**Branch base note.** The worktree was created from `a9aea282`, ~40 commits
behind `main` and predating the whole DeepSeek-V4 line — none of the files this
task names existed in it. The branch was fast-forwarded to `main` (`f2edd4cc`)
before any work; it had no commits of its own, so this was a clean
fast-forward, not a merge. Two untracked-in-git fixture files
(`artifacts/quality/instruction_hf_dolly48_prompts.jsonl`, the
`.superpowers/sdd/goal-objective/*.json` goldens) had to be copied in from the
shared checkout for collection to succeed; both directories are `.gitignore`d,
so nothing was committed.

---

## 1. The drafter forward

Reference: `.repos/omlx/omlx/patches/mlx_lm_mtp/deepseek_v4_dspark.py` for the
block, and `mlx_lm_mtp/deepseek_v4_model.py:228-264, 418-538` for the
whole-model entry points (which is where `main_proj`'s consumer, the tap
capture, and the noise-token block construction actually live — the dspark
module alone does not show them).

### `DSparkContextCache`

Physical ring of committed context K/V, one per stage, sized `sliding_window`.
Draft-block K/V is never committed to it, so **a rejected draft needs no
rollback of this cache at all** — it is always on the committed timeline. That
is a different question from `PoolingCache.is_trimmable`, which is about the
backbone's pooled cache; §4 is that one.

Once full, slots follow `absolute_position % max_size` and DSpark attends them
in that order rather than rotating back to chronological order.
`_chronological`/`_physical` exist only to splice an append into that ordering.
A non-contiguous append raises rather than silently mis-positioning.

### `DSparkAttention`

Subclasses `LocalAttention` for the projections and **overrides `__call__`
entirely**. Three reasons reuse was impossible rather than merely awkward:

* `mask=None`, deliberately. DSpark proposes a whole block in parallel, so slot
  `j` attends slot `j+1`. Under the backbone's causal mask that is forbidden.
  Non-causality is confined to the block; the committed context is all past.
* Keys come from `DSparkContextCache` concatenated with this block's own K/V,
  not from `RotatingKVCache.update_and_fetch`.
* `query_width` scores fewer positions than it keys against.

Output projection reuses `_project_attention_output`, which turned out to be
character-for-character what the reference inlines.

### `DeepseekV4FlashMTPBlock.__call__`

Hyper-connections → non-causal attention → (narrow to `output_width`) → MoE →
hyper-connections. The narrowing sits *after* attention and *before* the MoE,
which is the only placement where a narrower `query_width` saves anything, and
is asserted as such (§3).

`fp32=True` on the routed `LimitedSwiGLU` and `fp32_swiglu=True` on the shared
experts. **Enforced**: `DeepseekV4FlashVQMoE.bind_switch_mlp` now refuses a
routed module whose activation's `fp32` flag disagrees with the block's, in
both directions — a drafter stage refuses a bf16 activation and a backbone
layer refuses an fp32 one. The failure mode this guards is the same one the
increment-1 review found for the missing clamp: it binds, it runs, it stays
finite, and it is only visible in the acceptance rate.
`bind_deepseek_v4_flash_mtp_dense_experts` was building
`LimitedSwiGLU(limit)` — no `fp32` — and assigning `switch_mlp` directly,
bypassing the check. Fixed, and it now asserts the block agrees.

A drafter stage given a non-zero `compress_ratio` raises. The released
`compress_ratios` drafter tail is `[0, 0, 0]`; a compressed drafter stage would
be a config bug, not a supported variant.

### Model-level surface

```python
model.make_mtp_cache()                                   # one ring per stage
model.dspark_append_context(main_hidden, cache, *, start_offset=None)
model.dspark_draft_block(anchor_ids, cache, *, draft_length=None)
model.dspark_forward(main_hidden, anchor_ids, cache=None, *, draft_length=None)
model.dspark_markov(token_ids)                           # (bias logits, embedding)
model.dspark_confidence(head_hidden, markov_embedding)   # see the warning below
```

`dspark_forward` is the reference's single entry point. It is *split* into
append + draft because the teacher runner walks supervised positions forward
and appends one token between drafts; re-appending the whole prefix per
position would be quadratic.

`dspark_draft_block` returns `(logits, head_hidden)`: `head_hidden` is the last
stage's `hc_head` output, i.e. the exact input `norm` + `lm_head` consume.
That is what MTP-head recovery training regresses from, which is why it is
returned rather than left to be recomputed.

**`dspark_confidence` is wired from its shapes, and that is stated in its
docstring as a `.. warning::`.** `confidence_head.proj` is
`[1, hidden_size + dspark_markov_rank]`, which fixes the input as the final
hidden concatenated with the Markov embedding and the output as one scalar per
position. The *squashing* is not fixed by anything: **the reference constructs
this head and never calls it** — `confidence` has exactly one hit in the whole
omlx tree, the constructor at `deepseek_v4_dspark.py:326`. It returns the raw
score. Do not treat it as a calibrated acceptance probability until a reference
call site exists.

### Opt-in backbone capture

`DeepseekV4FlashBackbone.__call__(..., return_dspark_hidden=True)` returns
`(out, taps)` where `taps` is `_dspark_tap(h)` at each
`dspark_target_layer_ids` layer concatenated **in that list's order** — not
sorted order; `main_proj`'s columns were trained against the list. The tap is
`h.mean(axis=2)` over the hyper-connection streams
(`deepseek_v4_model.py:236`), *not* the `hc_head` collapse the LM head uses.

`_dspark_tap` is a module-level function precisely so the capture has one call
site that a test can count. Default off; the default branch is a separate loop
that computes no taps at all.

---

## 2. `--mode mtp-targets`

The refusal narrows from `MtpTargetsNotImplemented` ("the forward does not
exist") to `MtpDrafterUnavailable` ("this workload has no drafter, or its
stages have no routed experts bound"). The CLI no longer short-circuits the
mode before taking the lock.

### What it emits

Per supervised position `p`: context through `p`, anchor `token_ids[p]`,
proposals for `p+1 .. p+W`.

| array | dtype / shape | meaning |
| --- | --- | --- |
| `positions` | i32 `[P]` | supervised positions |
| `target_token_ids` | i32 `[P]` | next token, as in `logits` mode |
| `mtp_draft_width` | i32 scalar | `W` |
| `mtp_target_token_ids` | i32 `[P, W]` | tokens `p+1..p+W`, `-1` past the end |
| `mtp_target_valid` | bool `[P, W]` | False where the session ran out |
| `mtp_topk_logit_ids` | i32 `[P, W, K]` | drafter top-K ids |
| `mtp_topk_logit_values` | f16 `[P, W, K]` | their logits |
| `mtp_logsumexp` | f32 `[P, W]` | full-vocabulary normaliser |
| `mtp_tail_mass` | f16 `[P, W]` | mass outside the top-K |
| `mtp_final_hidden` | f16 `[P, W, H]` | drafter `hc_head` output |

Plus the same identity block every mode carries: `record_type`
`dsv4_teacher_mtp_targets_v1`, `prefill_chunk_tokens`,
`generation_config_sha256`, `token_ids_sha256`. Resume re-validates the file
and refuses a chunk mismatch, same as `logits`.

`(mtp_final_hidden, mtp_topk_*)` is a **complete recovery-training pair**: the
input side never has to be regenerated by re-running the 163 GB teacher.
`test_mtp_targets_hidden_and_logits_are_a_usable_recovery_pair` proves it by
pushing the stored hidden back through the real `norm` + `lm_head` and
recovering the stored top-K exactly — storing the wrong hidden (the pre-
`hc_head` 4D stream, say) would be finite, the right rank, and useless.

### Costs, stated up front

* **Compute.** The context ring is appended forward once (positions ascend), so
  context work is linear in the session. But each position runs a `W`-wide
  block through every drafter stage: `3 × W` token-layers per supervised
  position against the backbone's 43 — roughly a tenth of the backbone pass at
  the released shapes.
* **Disk.** The artifact is `W` times a logits-mode one. `--mtp-draft-width`
  (default `dspark_block_size` = 5) is the knob if that binds.
* **Memory.** `load_dsv4_streaming_workload(with_mtp=True)` binds the drafter's
  routed experts **dense-resident**, ~13 GB/stage in bf16. Resident and not
  streamed because the drafter is re-entered once per supervised position;
  streaming would re-read 3.4 GB thousands of times per session instead of the
  43 reads the backbone pays. This is the one place I knowingly took the
  expensive option — see concern 2.

---

## 3. Test summary

```
tests/test_deepseek_v4_flash_adapter.py   73 passed  (56 before, 17 new)
tests/test_dsv4_teacher_runner.py         56 passed  (49 before, 8 new,
                                                      1 refusal test rewritten)
                                         129 passed in ~23 s
```

Blast radius, unchanged and green: `test_family_registry` 13,
`test_deepseek_v4_policy` 25, `test_fp4_expert_dequant` 68,
`test_fp8_block_dequant`, `test_glm45_air_vq_adapter` 15,
`test_glm52_adapter_binding` 5, `test_switch_routing`,
`test_quantized_vq_switch_linear` — **171 passed** together.

Ruff parity in every file touched: adapter + its tests 29 findings before, 29
after; runner + its tests + the CLI 2 before, 2 after.

The four assertions the brief asked for by name, and how each got teeth:

**fp32 is actually fp32.** Two independent proofs. A recorder monkeypatched
over the module-level `_limited_swiglu` shows every drafter SwiGLU — routed and
shared, all stages — receives `float32` operands. Then, because a dtype alone
proves nothing about the answer, the same drafter is run in bfloat16 twice,
once with the promotion and once without: the same path twice is `array_equal`
(so the gap is the promotion and nothing else), and the fp32-vs-bf16 gap
exceeds `1e-3 × scale`.

**Non-causality is real, asserted on mask semantics not shapes.** Perturb only
the *last* block position's input; the *first* query position's output must
move. Under a causal mask it could not. The control runs the identical
perturbation through the backbone's `LocalAttention`, where position 0 comes
out `array_equal` and the last position does move — so the test discriminates
rather than merely passing.

**`main_proj` pulls exactly the target layers.** On a synthetic config with
`dspark_target_layer_ids = [3, 1]` — deliberately out of ascending order and
skipping layer 2 — the two tap slices are `array_equal` to independently
recomputed layer-3 and layer-1 hidden states, and are asserted *not* equal to
any of the other three layers or to each other's positions. So a capture that
sorted the ids, or grabbed `[0, 1]`, fails rather than producing a
right-shaped wrong answer. For the real config: `[40, 41, 42]`, asserted as the
last three of the 43-layer backbone, `main_proj_in_features == 12288`, and the
built model's `main_proj.weight` is `(4096, 12288)`. The synthetic 2-target
case is the same test's fixture.

**Capture is opt-in and the default is unchanged.** The default forward returns
a bare `mx.array` (not a tuple), its logits are `array_equal` to the opted-in
path's, and the tap call counter is **exactly zero** on the default path and
exactly `len(dspark_target_layer_ids)` when opted in, with the recorded shapes
asserted. `mx.get_peak_memory()` is deliberately *not* the instrument: MLX
reuses freed blocks across runs, so the second forward of a pair can peak lower
whatever it computes (measured on the fixture: plain 3.01 MB, capture 2.94 MB,
in that order). The assertion and the measurement are both in the test's
comment so nobody re-adds the flaky check.

**Drafter forward correctness in the runner.**
`test_mtp_targets_capture_matches_a_per_position_reference` re-runs a
from-scratch `dspark_forward` over the whole prefix at every supervised
position and asserts the top-K ids and final hiddens match the forward-walked
incremental capture — which is the only thing that makes the linear-time
context walk sound.

---

## 4. `is_trimmable()` — **implemented, not deferred**

**Decision: implemented, with the undo log, here.** Reasoning: the log is
self-contained (~60 lines confined to `PoolingCache`), it is directly testable
without any verify runtime, and its only dependency on stripped omlx machinery
was `cache_rollback._is_undo_armed()` — a *performance* gate deciding whether
to stash, not a correctness input. Deferring it would have been the
"known win but deferred" shape: the increment-1 report already named it as a
Wave-5 blocker, and nothing in Wave 5 would have made it easier.

What it does. A trim of `n` is free while those tokens are still in the
partial-window remainder buffer. The hard case is the token that *completed* a
window: its `ratio` predecessors were consumed out of `buf_kv`, compressed and
appended, and the remainder no longer holds them. `accumulate_windows` now
stashes — before it mutates anything — the pre-update remainder rows, the
pre-update pooled length and overlap windows, and this update's raw `(kv,
gate)`. `trim` replays that prefix minus `n`.

`is_trimmable()` returns `True` in exactly two situations: nothing pooled yet,
or the last token is recoverable (still in the remainder, or the undo log holds
it). A `trim` it cannot cover returns **0 and changes nothing** — a partial
trim would leave the pooled run and the local KV cache describing different
timelines, which is the corruption the `False` was guarding against.

Two deliberate divergences from `cache_extras.py`, both in the docstring:

* the stash records the pre-update pooled **length**, not the pooled array.
  Upstream's `trim` only ever reads `pooled_prev.shape[1]`, and `self.pooled`
  is a lazy view of `_pool_buf` that later in-place appends could alias.
* there is no `_undo_chain`. Upstream chains consecutive decode updates into
  one record, gated on omlx arming flags stripped with the verify machinery.
  Without them the honest contract is one update deep, and `trim` past that
  returns 0 rather than guessing.

Proof, not assertion-of-presence — four tests:

* `..._trim_restores_exact_prior_state` runs to the hard case on purpose (the
  8th token completes the 2nd window) and compares a full materialised
  snapshot — remainder, pooled length, buffer rows, pooled rows, both overlap
  windows — for exact equality via `np.array_equal`.
* `..._trim_then_replay_reaches_the_clean_state` takes a *wrong* 8th token,
  rolls it back, then replays the accepted 8th and 9th, and asserts the result
  is state-identical to a cache that never saw the wrong one. No residue.
* `..._refuses_a_trim_it_cannot_cover` asserts `trim(2) == 0`, that nothing
  changed, and that refusing did not consume the record.
* `..._reports_trimmable_only_when_it_can_actually_undo` walks all three
  answers, including the honest `False` after a prompt-sized update that
  divides evenly.

---

## 5. Concerns / carry-forward

1. **The drafter forward has no numerical reference.** It runs, it is finite,
   it responds to its inputs, and its structure is asserted against the
   reference line by line — but there is no logit parity against a running
   DSpark, because no runnable reference implementation of this architecture
   exists locally (the same limitation increment 2 recorded for the backbone).
   Everything here is "the reference's expression, transcribed and exercised",
   not "measured equal to the reference's output". Acceptance-rate parity is
   the first Wave-5 measurement that would catch a transcription error.

2. **The drafter's routed experts are dense bf16 resident: ~13 GB/stage,
   ~39 GB for three.** That fits alongside the ~9 GB of backbone residents on a
   128 GB machine, but it is 4x what it needs to be. The native-mxfp4 path
   already exists for the backbone (`_MXFP4SwitchLinear` + the span index) and
   would put the drafter at ~3.4 GB/stage; it needs
   `build_dsv4_expert_span_index` generalised past its hardcoded `layers.`
   prefix (the `mtp.K.ffn.experts.` names parse identically under the same
   `name.split(".")[1]`). This is an optimisation, not a correctness gap, and
   it is flagged in `load_dsv4_streaming_workload`'s docstring — but it should
   land before the real `mtp-train` run, because 39 GB of avoidable residency
   is exactly the kind of cost that turns into a thermal/paging problem at
   hour 20.

3. **`dspark_confidence`'s activation is unpinned.** See §1. The head is
   constructed by the reference and never called anywhere in omlx. It is
   wired, it is finite, and it returns a raw score with a docstring warning.
   Wave 5 must find or decide the squashing before any acceptance policy reads
   it.

4. **`mtp_target_capture` supervises `dspark_block_size` slots at every
   supervised position, which is a design choice not a derivation.** The
   alternative — slot 0 only — would make the mode a second logits mode and
   throw away the block structure MTP training exists to learn. The choice is
   `--mtp-draft-width`-adjustable and stamped into the generation config sha,
   so a corpus produced at one width cannot be silently mixed with another.
   But nobody has yet stated what width the *recovery trainer* wants; if it
   turns out to want 1, the runs are 5x oversized.

5. **The `self.dspark` exact-attention decode branch is still stripped**
   (carried forward from increments 1 and 2, unchanged). Decode is not
   bit-identical to upstream until Wave-5 verify brings it back.

6. **No real run was launched, by instruction.** The mode is tested end to end
   on a tiny synthetic model with a 2-stage drafter over layers `[2, 3]`,
   including the resume path and the chunk-mismatch refusal. The first real
   `mtp-train` session will be the first time the drafter sees released
   weights, and the first thing to check is that the drafter's own logits are
   not degenerate (`validate_dsv4_mtp_targets_capture` refuses a constant
   `mtp_final_hidden`, which catches the loudest version of that).

7. **Branch hygiene — merge is clean.** This branch is `f2edd4cc` + three
   commits. `main` advanced to `dded37e0` (the sibling agent's Wave-5 VQ
   materialization pilot) while this ran; that commit adds three files and
   touches **none** of the six here, so there is no conflict to resolve.

   The untracked fixture copies described at the top are working-tree-only and
   will not follow the merge: whoever runs the full suite on a fresh worktree
   will hit the same collection errors (`artifacts/quality/` and
   `.superpowers/sdd/goal-objective/` are both `.gitignore`d) and needs the
   same copies. Neither is caused by this work, and neither affects the two
   suites this increment owns.

---

# Adversarial review — fixes

Reviewer verdict: drafter forward, tests and undo log **pass with teeth**; one
HIGH and two MEDIUMs. All three are fixed. The HIGH was a real train/inference
skew that my own tests could not see, because the test had copied the bug.

| SHA | Subject |
| --- | --- |
| `000c0f25` | `fix(dsv4): the DSpark context seam, the trim contract, and mtp-targets cost` |

## HIGH — the mtp-targets context seam was off by one

`mtp_target_capture` committed the fused taps through position `p`
**inclusive** and then drafted with anchor `token_ids[p]`. The reference
commits through `p-1`:

* `deepseek_v4_model.py:540-554` — `dspark_calibration_forward` uses
  `context = target_hiddens[:, :-1]` with `anchor = input_ids[:, -1:]`;
* `dspark.py:204-209` — `take_primed` **refuses** unless the ring is exactly
  one token behind the target cache.

Two consequences, neither visible as a crash or a NaN. Every draft slot's RoPE
position was shifted `+1` relative to the Wave-5 verify runtime, so a head
trained on these targets would be trained at the wrong positions. And the
drafter was conditioned on the anchor's *own* backbone hidden state —
information it structurally cannot have at inference, which is the classic
teacher-forcing leak: it would have scored well in training and degraded on
the first real speculative step.

**Why my tests missed it, and what changed.** The runner test built its
"reference" by hand-slicing `fused[:, :position + 1]` into `dspark_forward` —
the same off-by-one, so it compared the bug against itself. The fix is
structural, not a tightened tolerance:

* `DeepseekV4FlashVQModel.dspark_calibration_forward` is now a 1:1
  transcription of the reference method. It takes a **whole prefix** (taps and
  ids together, lengths cross-checked) and does its own slicing, so no caller
  can express the seam wrongly. Its docstring states the invariant and both
  failure modes.
* `mtp_target_capture` walks `fused[:, committed:position]` with
  `committed = position`.
* `test_mtp_targets_capture_matches_the_reference_calibration_forward` hands
  the reference the whole prefix and lets *it* slice. The reference is the
  reference.
* Two new tests assert the seam directly rather than by consequence:
  `test_calibration_forward_excludes_the_anchors_own_tap` (adapter) and
  `test_mtp_targets_context_excludes_the_anchors_own_tap` (runner). Both check
  the ring offset is exactly `p`, **and** that the off-by-one variant runs, is
  finite, and produces materially different logits — so the equality is a
  statement about the seam and not a tautology.

## MEDIUM 1 — `trim(n > 1)` could desync silently

`is_trimmable()` takes no `n`, and `trim` returned `0` for an uncoverable
request. `CacheList.trim` returns only its **last** sub-cache's result, so a
Wave-5 block rollback — `trim(block_size - (accepted + 1))`, i.e. `n > 1` as
the normal case — would have moved `RotatingKVCache` by `n`, left
`PoolingCache` at 0, returned `n` to the caller, and carried two different
timelines forward with nothing to signal it.

* `trim` now **raises** `ValueError` naming `n`, the remainder, what the undo
  log covers, and `max_undo_update`. `trim(0)` is still a no-op; the `n == 1`
  consumer contract is unchanged. The docstring says why this diverges from
  both `cache_extras.py` and mlx-lm's usual "return what you managed".
* `_POOLING_UNDO_MAX_UPDATE = 8` — a constant that merely *happened* to exceed
  `dspark_block_size + 1 == 6` — is gone. `deepseek_v4_flash_pooling_undo_window(config)`
  returns `dspark_block_size + 1` (the widest single update a verify cycle can
  produce: the block plus its anchor), `make_cache()` threads it into every
  `PoolingCache`, and a bare `PoolingCache` defaults to **1** — decode only,
  because a cache built without a config cannot honestly promise more.
* Three tests: the raise (with state proved unchanged and the record not
  consumed), the derived window across four configs including the real one and
  a `dspark_block_size=0` fallback, and a block-sized update that is undoable
  with the derived window and refused with the decode-only default.

## MEDIUM 2 — `mtp_target_capture` RAM and disk honesty

* fp16 at capture, not at the final stack. `mtp_topk_logit_values`,
  `mtp_tail_mass` and `mtp_final_hidden` are fp16 in the schema anyway; holding
  them as float32 for a whole session doubled the transient. At ~12.5 k
  supervised positions, `W=5`, `K=2048`, `H=4096`: **2.0 GB → 1.3 GB per
  session**, which is now the artifact size itself, the floor.
* The docstring's disk estimate had omitted `mtp_final_hidden` — half the
  total. It is now a table with every term's arithmetic: ~102 KB per supervised
  position, **~1.3 GB per session, ~154 GB for the 120-session `mtp-train`
  split**, with a note that `--mtp-draft-width` scales all of it linearly and
  `top_k` scales the two logit terms only.
* `mx.clear_cache()` left the per-position hot loop (12.5 k calls per session).
  Every draft block allocates identical shapes, so clearing per position threw
  away precisely the buffers the next iteration wanted; it now runs every 512
  positions and once at the end.

## One test assertion corrected, not weakened

`test_mtp_targets_hidden_and_logits_are_a_usable_recovery_pair` had asserted
exact **rank** agreement between logits replayed from the stored fp16 hidden
and the stored top-K (computed from the fp32 hidden). Two entries a hair apart
can legitimately swap deep in the top-K, and after the seam fix they did. It
now asserts what the recovery pair actually claims — the logit **values** at
the stored ids agree to fp16 tolerance, and the argmax (which has a real
margin) survives exactly — plus a control showing a different hidden does *not*
reproduce them, compared against the replay gap rather than an absolute
epsilon this fixture's O(0.05) logits would make arbitrary.

## Post-fix test summary

```
tests/test_deepseek_v4_flash_adapter.py   76 passed  (73 before the fixes)
tests/test_dsv4_teacher_runner.py         57 passed  (56 before the fixes)
                                         133 passed in ~5 s
```

Ruff parity across the four source/test files: **31 findings before, 31
after** (the CLI file has none, before or after).
