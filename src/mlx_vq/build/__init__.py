"""Declarative build recipes for KEEP artifacts.

A recipe (one YAML per candidate under ``recipes/``) declares an ordered
graph of steps over the existing deterministic primitives. The runner
content-addresses each step, reuses completed steps, and records the full
lineage as ``build_steps`` in the output artifact manifest.
"""
