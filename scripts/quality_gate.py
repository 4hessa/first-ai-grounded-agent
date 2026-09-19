"""Portable offline quality gate for the portfolio repository."""
from __future__ import annotations

import compileall
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def verify_packaged_defaults() -> bool:
    packaged = ROOT / "first_ai" / "data"
    try:
        if json.loads((ROOT / "config.json").read_text(encoding="utf-8")) != json.loads(
            (packaged / "config.json").read_text(encoding="utf-8")
        ):
            print("Packaged config is out of sync with config.json.", file=sys.stderr)
            return False
        source_files = sorted(path.name for path in (ROOT / "knowledge").glob("*.md"))
        packaged_files = sorted(path.name for path in (packaged / "knowledge").glob("*.md"))
        if source_files != packaged_files:
            print("Packaged knowledge file list is out of sync.", file=sys.stderr)
            return False
        for name in source_files:
            if (ROOT / "knowledge" / name).read_bytes() != (packaged / "knowledge" / name).read_bytes():
                print(f"Packaged knowledge file is out of sync: {name}", file=sys.stderr)
                return False
    except (OSError, ValueError):
        print("Could not validate packaged default data.", file=sys.stderr)
        return False
    return True


def main() -> int:
    print("[1/3] Compiling first_ai...")
    if not compileall.compile_dir(ROOT / "first_ai", quiet=1):
        print("Compilation failed.", file=sys.stderr)
        return 1

    print("[2/3] Verifying packaged default data...")
    if not verify_packaged_defaults():
        return 1

    print("[3/3] Running deterministic test suite...")
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
