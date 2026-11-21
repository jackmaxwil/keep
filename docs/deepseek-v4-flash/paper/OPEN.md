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
| Head-to-head at 64, 96 and 128 GB against other runtimes | the paper's goal (Jack, 2026-11-20) is to show this system ahead of oMLX and other Apple-silicon runtimes on budget hardware; today the comparison is indirect. Checked on 2026-11-21 from the Hugging Face file lists: the smallest oMLX set is 313 GB and ds4's is 366 GB, neither fits this Mac's disk; llama.cpp with the IST-DASLab GSQ-RCO Coder build (192 of 384 experts, 72.1 GB of weights and a 110.6 GB Engram table, 182.7 GB) does, and Jack lifted the download rule that day. In progress: that build under llama.cpp on this Mac (`jobs/llamacpp_mac.sh`: same request, sampling and scorer as our rows; tokens per second, footprint, scripts correct); its kept-experts list as a static set at our codes in the simulator (`jobs/dialkl_mac.sh`, variant `ista192`); our rows at 73, 140 and 207 residents under re-route, selective wait and full wait (`jobs/dial_mac.sh`) | results; a KL for the llama.cpp row needs its logits on our texts (`llama-perplexity` or the server's `n_probs`); the 64 and 96 GB machines are emulated on the 128 GB Mac by wiring 64 or 32 GiB (`scripts/v41_build/ballast.py`, approved by Jack on 2026-11-21), which takes the memory away but leaves the GPU working-set limit and `hw.memsize` of the larger machine; oMLX and ds4 need about 130 GB more free disk |
| Simulator matched to the served timing | two referee reviews (2026-11-21): quality comes from a simulator that installs after one token and adapts inside a prompt, speed from a system that does neither; their agreement was never measured | `lag:N` and `frozen:P` in the simulator, then Table 2 rerun |
| Generation with more evidence | one prompt; the static row ran with the speculative decoder on; no full-model count; 8 samples at 96 and 120 residents | static 144 with the decoder off, 24 samples at 96 and 120, more prompts; a full-model count needs a machine that holds it |
| Speed, repeated | ranges cover two or three requests beside a 25% run-to-run spread; the long-prompt request has no static counterpart | interleaved repeats of static and routed at one threshold, long prompt included |
| A selective wait rule | the closest prior work finds that waiting for the strongest misses dominates re-routing every miss | simulator (`wait:W`, queued) and serving (`OMLX_ROUTED_WAIT_MIN`) |
