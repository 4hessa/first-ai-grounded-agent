"""Portable offline quality gate for the portfolio repository."""
from __future__ import annotations

import compileall
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    print("[1/2] Compiling first_ai...")
    if not compileall.compile_dir(ROOT / "first_ai", quiet=1):
        print("Compilation failed.", file=sys.stderr)
        return 1

    print("[2/2] Running deterministic test suite...")
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
