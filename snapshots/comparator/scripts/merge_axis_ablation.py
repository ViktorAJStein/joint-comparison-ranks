#!/usr/bin/env python3
"""Merge per-configuration axis-GMM CSV files and annotate their parameters."""

from __future__ import annotations

import argparse
import csv
import glob
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pattern", help="Glob containing _dD_sS_deltaDELTA.csv files")
    parser.add_argument("output", type=Path)
    parser.add_argument("--J", type=int, default=None)
    args = parser.parse_args()

    pattern = re.compile(r"_d(\d+)_s(\d+)_delta([0-9.]+)\.csv$")
    rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None
    sources = sorted(glob.glob(args.pattern))
    if not sources:
        raise SystemExit(f"No files match {args.pattern!r}")

    for source in sources:
        match = pattern.search(source)
        if match is None:
            raise ValueError(f"Cannot infer d, s, delta from {source}")
        d, sparse_dims, delta = match.groups()
        with open(source, newline="") as handle:
            reader = csv.DictReader(handle)
            if fieldnames is None:
                fieldnames = list(reader.fieldnames or []) + ["sparse_dims", "delta_param"]
                if args.J is not None:
                    fieldnames.append("J")
            for row in reader:
                row["sparse_dims"] = sparse_dims
                row["delta_param"] = delta
                if args.J is not None:
                    row["J"] = str(args.J)
                rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows from {len(sources)} files to {args.output}")


if __name__ == "__main__":
    main()
