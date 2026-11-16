# Open before posting

Build: `tectonic paper.tex` (or `pdflatex`, `bibtex`, `pdflatex` twice) in this directory. arXiv
does not run BibTeX, so upload `paper.bbl` with the source. Figures: `uv run --with matplotlib
python figures.py`. Numbers: `python stats.py > stats.md` (needs the stored evals).

| Item | Why | Needs |
|---|---|---|
| Author block | affiliation and contact are not filled in | Jack |
| Total-mass 192 at 4 bits on the 22 sessions | separates expert choice from precision in the pruned-checkpoint comparison (done at 2 bits only) | 4-bit weights of the removed experts (pod) |
| Static sets in generation | the routed-against-static result has no task-level check | serving runs on the Mac, about 20 minutes each |
| A second prompt set for generation, and a full-model count | one prompt, 24 samples; no full-model baseline | design; a machine that holds the full model |
| Held-out confirmation of the settings | tau, hot set sizes and the skipping threshold were chosen on the evaluation text | pod: 54 untouched sessions, full length |
| Simulator with the served install delay and a fixed set inside a prompt | the simulator installs after one token | a simulator option and one rerun |
| A second MoE, ideally a narrow pool as well | every result is for one model; the prediction for narrow pools is untested | a download (Jack's go) |
| Table 1 values | expert counts for Mixtral, Qwen3 and DeepSeek-V3 need checking against the papers | a read |
| The wait rule in our own code path at 2 bits | the speed baseline is another runtime on another checkpoint | implementation |
| Prior-art pass on the three novelty statements | related work was checked against 15 papers only | a search |
