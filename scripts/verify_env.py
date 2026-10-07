#!/usr/bin/env python3
from __future__ import annotations

import json
import sys

from keep import verify_environment


def main() -> int:
    report = verify_environment()
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
