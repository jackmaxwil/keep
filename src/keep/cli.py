"""Console entry point for the ``keep`` command.

New user-facing surface only — the implementation lives in
``mlx_vq.build.cli`` until the GLM-5.2 cutover moves real code into this
package.
"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    from mlx_vq.build.cli import main as build_main

    return build_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
