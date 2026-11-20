# Open before posting

Build: `tectonic paper.tex` (or `pdflatex`, `bibtex`, `pdflatex` twice) in this directory. arXiv
does not run BibTeX, so upload `paper.bbl` with the source. Figures: `uv run --with matplotlib
--with numpy python figures.py`. Numbers: `python stats.py > stats.md`, `python stats.py --domains >
domains.json` and `python routing.py > routing.json` (they need the stored evals and routes);
`python stats_glm.py > stats-glm.md` (the second model, from `docs/glm-5.3-flash/results`).
The three listings are excerpts of `scripts/runpod/v41_reap/local_eval.py` (Listing 1),
`scripts/v41_build/omlx_routed.py` (`install`, `evict`) and `scripts/v41_build/omlx_lattice.py`
(`route`, `unroute`, and the kernel of Listing 3); keep them in step with the code.

| Item | Why | Needs |
|---|---|---|
| Author block | affiliation and contact are not filled in | Jack |
| Total-mass 192 at 4 bits on the 22 sessions | separates expert choice from precision in the pruned-checkpoint comparison (done at 2 bits only) | 4-bit weights of the removed experts (pod) |
| A second prompt set for generation, and a full-model count | one prompt, 24 samples; no full-model baseline | design; a machine that holds the full model |
| Held-out confirmation of the settings | tau, hot set sizes and the skipping threshold were chosen on the evaluation text | pod: 54 untouched sessions, full length |
| Simulator with the served install delay and a fixed set inside a prompt | the simulator installs after one token | a simulator option and one rerun |
| The rest of the order on the second model | GLM-5.3-Flash was measured for residency only (routed against static at native precision, done 2026-11-20); codes, the residents-against-bits trade, generation and serving are untested there, and its threshold of 0.2 was chosen after the fixed run | a code fit for 288 experts in 42 layers (pod), then the simulator |
| The wait rule in our own code path at 2 bits, with a selective wait (first choice or strong gate weight) | the speed baseline is another runtime on another checkpoint; Son (2026) finds a selective wait far cheaper in perplexity than re-routing every miss | implementation |
| CommitMoE (AAAI 2026) and the MLX-Flash package | the prior-art pass of 2026-11-16 could not open either; both may bear on serving a miss without waiting | reading them |
