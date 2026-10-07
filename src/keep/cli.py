"""Console entry point for the ``keep`` command.

The implementation lives in ``keep.build.cli``.
"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    from keep.build.cli import main as build_main

    return build_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
