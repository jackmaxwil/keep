# Open before posting

Build: `tectonic paper.tex` (or `pdflatex`, `bibtex`, `pdflatex` twice) in this directory. arXiv
does not run BibTeX, so upload `paper.bbl` with the source. Figures: `uv run --with matplotlib
--with numpy python figures.py`. Numbers: `python stats.py > stats.md`, `python stats.py --domains >
domains.json` and `python routing.py > routing.json` (they need the stored evals and routes);
`python stats_heldout.py > stats-heldout.md` (the held-out texts) and
`python stats_glm.py > stats-glm.md` (the second model, from `docs/glm-5.3-flash/results`).
The three listings are excerpts of `scripts/runpod/v41_reap/local_eval.py` (Listing 1),
`scripts/v41_build/omlx_routed.py` (`install`, `evict`) and `scripts/v41_build/omlx_lattice.py`
(`route`, `unroute`, and the kernel of Listing 3); keep them in step with the code.

| Item | Why | Needs |
|---|---|---|
| Author block | affiliation and contact are not filled in | Jack |
| Total-mass 192 at 4 bits on the 22 sessions | separates expert choice from precision in the pruned-checkpoint comparison (done at 2 bits only) | 4-bit weights of the removed experts (pod) |
| A second prompt set for generation, and a full-model count | one prompt, 24 samples; no full-model baseline | design; a machine that holds the full model |
| Held-out check of the precision results | the held-out run (2026-11-20) confirmed routed against static only; the hot set, the skipping threshold and the residents-against-bits crossover were not rescored on held-out text, and sessions were capped at 8,192 tokens | simulator runs on the Mac, about 10 h for the same 42 sessions |
| Simulator with the served install delay and a fixed set inside a prompt | the simulator installs after one token | a simulator option and one rerun |
| The rest of the order on the second model | GLM-5.3-Flash was measured for residency only (routed against static at native precision, done 2026-11-20); codes, the residents-against-bits trade, generation and serving are untested there, and its threshold of 0.2 was chosen after the fixed run | a code fit for 288 experts in 42 layers (pod), then the simulator |
| The wait rule in our own code path at 2 bits, with a selective wait (first choice or strong gate weight) | the speed baseline is another runtime on another checkpoint; Son (2026) finds a selective wait far cheaper in perplexity than re-routing every miss | implementation |
| CommitMoE (AAAI 2026) and the MLX-Flash package | the prior-art pass of 2026-11-16 could not open either; both may bear on serving a miss without waiting | reading them |
| Head-to-head at 64, 96 and 128 GB against other runtimes | the paper's goal (Jack, 2026-11-20) is to show this system ahead of oMLX and other Apple-silicon runtimes on budget hardware; today the comparison is indirect. A survey on 2026-11-20 (not re-verified) found no published speed or quality for this model at 128 GB or below, and three configurations to measure: oMLX 0.8.0 at its 2-bit format with expert offload, ds4 `ds41f-q2` with SSD streaming, llama.cpp with Unsloth's UD-IQ1_S. Same model, machine, prompts and texts; columns: fits, tokens per second, KL to the full model, scripts correct | the other runtimes' weights (233 to 341 GB each) on an Apple-silicon machine; Jack ruled out downloads to the Mac on 2026-11-20, so this is blocked. Until then: a wait rule in our own code path on the codes already on disk, as the exact-offload baseline |
