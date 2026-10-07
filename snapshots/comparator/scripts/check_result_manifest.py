#!/usr/bin/env python3
"""Write or verify SHA-256 hashes for retained result data and figures."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MANIFEST = RESULTS / "manifest.sha256"


def retained_files() -> list[Path]:
    paths: list[Path] = []
    for directory in (RESULTS / "data", RESULTS / "figures"):
        if directory.exists():
            paths.extend(path for path in directory.rglob("*") if path.is_file())
    return sorted(paths)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rendered_manifest() -> str:
    return "".join(
        f"{digest(path)}  {path.relative_to(ROOT).as_posix()}\n"
        for path in retained_files()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    actual = rendered_manifest()
    if args.write:
        MANIFEST.write_text(actual)
        print(f"wrote {MANIFEST.relative_to(ROOT)} with {len(retained_files())} entries")
        return
    expected = MANIFEST.read_text()
    if actual != expected:
        raise SystemExit(
            "Result manifest mismatch. Inspect changed data and run "
            "`uv run python scripts/check_result_manifest.py --write` only if intentional."
        )
    print(f"verified {len(retained_files())} retained result files")


if __name__ == "__main__":
    main()
