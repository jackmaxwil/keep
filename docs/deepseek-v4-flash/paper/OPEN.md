# Open before posting

Build: `tectonic paper.tex` (or `pdflatex`, `bibtex`, `pdflatex` twice) in this directory. arXiv
does not run BibTeX, so upload `paper.bbl` with the source. Figures: `uv run --with matplotlib
--with numpy python figures.py`. Numbers: `python stats.py > stats.md`, `python stats.py --domains >
domains.json` and `python routing.py > routing.json` (they need the stored evals and routes).
The three listings are excerpts of `scripts/runpod/v41_reap/local_eval.py` (Listing 1),
`scripts/v41_build/omlx_routed.py` (`install`, `evict`) and `scripts/v41_build/omlx_lattice.py`
(`route`, `unroute`, and the kernel of Listing 3); keep them in step with the code.

| Item | Why | Needs |
|---|---|---|
| Author block | affiliation and contact are not filled in | Jack |
| Total-mass 192 at 4 bits on the 22 sessions | separates expert choice from precision in the pruned-checkpoint comparison (done at 2 bits only) | 4-bit weights of the removed experts (pod) |
| Static sets in generation | the routed-against-static result has no task-level check | serving runs on the Mac, about 20 minutes each |
| A second prompt set for generation, and a full-model count | one prompt, 24 samples; no full-model baseline | design; a machine that holds the full model |
| Held-out confirmation of the settings | tau, hot set sizes and the skipping threshold were chosen on the evaluation text | pod: 54 untouched sessions, full length |
| Simulator with the served install delay and a fixed set inside a prompt | the simulator installs after one token | a simulator option and one rerun |
| A second MoE | every result is for one model; GLM-5.3-Flash (288 experts, 8 active; this repo already has an adapter and a plan for it), GLM-5.3 or Kimi K3.1 Pro are the current candidates, and a narrow pool would test where the order should fail | a download (Jack's go) |
| The wait rule in our own code path at 2 bits | the speed baseline is another runtime on another checkpoint | implementation |
| Prior-art pass on the three novelty statements | related work was checked against 15 papers only | a search |
| License | the paper links the public repository (title footnote, a Code availability section, a link beside each listing; all point at `main`), but the repository has no license file | Jack |
