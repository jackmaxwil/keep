Break this paper before a referee does.

The paper is `docs/deepseek-v4-flash/paper/paper.tex` in `/Users/jackmazac/Development/keep`
(built: `paper.pdf`, 31 pages). Title: *Pruning Is a Frozen Cache: Unifying Pruning, Quantization
and Offloading in Mixture-of-Experts Inference*. It is headed for arXiv.

## Where it stands

- One model carries almost everything: DeepSeek-V4.1-Flash-1017, 40 layers, 384 experts per layer,
  6 active, served on one M4 Max with 128 GB through a third-party runtime (oMLX) that the code
  patches at load time.
- Quality is KL to the full model from a simulator: 17 texts of 1,024 tokens on which the settings
  were chosen, and 36 held-out texts with a median of 213 tokens.
- Generation is one coding prompt, 24 samples a row at best, no row for the full model.
- A second model, GLM-5.3-Flash, tests one comparison only, under forced context, with no codes.
- Speed is two or three requests per configuration, beside a 25% run-to-run spread.
- The abstract now says: "We do not compare with other runtimes at equal memory."

## What the author wants from the paper

In his words:

> i want our paper to show KEEP and RAMP are better than oMLX and competitor MLX inference
> frameworks for running massive MoEs on more budget constrained hardware (less memory, less
> FLOPS, etc) while remaining competitive in accuracy and speed

KEEP is the compression code (`src/keep`), RAMP the runtime code (`src/ramp`); the paper's serving
code is in `scripts/v41_build`, its simulator in `scripts/runpod/v41_reap`, the second model's run
in `scripts/runpod/glm53`.

Read that sentence as three claims a reader could check: better than a named runtime, on a
smaller machine, without giving up accuracy or speed. The paper as written supports none of the
three head on. The distance between that sentence and the paper is the thing to map.

## What to do

Attack it. Nothing in the paper is protected: the title, the framing, the formulation, the
metric, the choice of baselines, whether the central result is interesting at all, whether the
right paper is a different one. Find what a hostile but competent reader would find, and then
find what they would miss.

Start from more than one seat and let them disagree:

- a referee at a systems venue who has reviewed five expert-offloading papers this year
- the maintainer of oMLX, or of llama.cpp, reading how their runtime is described
- someone with a 64 GB Mac deciding tonight whether this is worth installing
- a statistician who trusts nothing with n = 17
- one seat of your own choosing that none of these would think of

Do not take the paper's word for its numbers. The tables are generated: `stats.py`,
`stats_heldout.py` and `stats_glm.py` beside the paper print them, the second model's per-text
results are in `docs/glm-5.3-flash/results`, and the first model's stored evaluations are in
`~/Models/v41-reap-run/evals`. The simulator's policy is `residual_cache` in
`scripts/runpod/v41_reap/local_eval.py`; the served policy is `scripts/v41_build/omlx_routed.py`
and `omlx_lattice.py`. Where the paper describes its own code, check the code. Where it describes
someone else's work, check the source on the web.

Three things you cannot know from the files. The Mac is running a queue of long jobs: start no
model, no server and no simulator run, and keep any process under a few gigabytes. Nothing may be
downloaded to this machine. And this is a review: change no file, commit nothing.

The paper was revised once already after two reviews. Do not read that history first. When your
own list is finished, read `git show df2f1dc` and `docs/deepseek-v4-flash/paper/OPEN.md`, and
sort your findings into those already known and those that are new. The new ones are what this
is for.

## What to hand back

1. **Attacks, ranked by how much damage they do.** For each: the claim, quoted, with its line in
   `paper.tex`; why it fails; the evidence, as a file and line, a number you recomputed, or a
   URL; whether the earlier reviews already had it; and the smallest thing that would rescue the
   claim, as a rewording or as one experiment.
2. **Errors of fact.** Any number, citation or description of code or of another system that is
   wrong, with the correct value and where you found it.
3. **The paper this evidence supports.** A title and an abstract of at most 150 words that claim
   only what the files show today. If that paper is not worth posting, say so and say why.
4. **The shortest road to the author's sentence.** The experiments that would let the paper make
   his three claims honestly, in order, each with what it needs (a GPU pod and a 380 GB volume
   holding GLM-5.3-Flash are available; other runtimes' weights cannot come to this Mac), what
   result would count as a win, and what result would mean the claim is false.
5. **One thing the author is not going to want to hear**, stated plainly.
