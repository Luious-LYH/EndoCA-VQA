"""Prepare the bundled EndoCA manifests for evaluation."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path


SUITES = {
    "core": ("endoca_core.jsonl.gz", "endoca_core.jsonl", 27736),
    "diagnostic": ("endoca_diagnostic.jsonl.gz", "endoca_diagnostic.jsonl", 15300),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_manifest(source: Path, destination: Path, force: bool) -> None:
    if destination.exists() and not force:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(source, "rb") as src, destination.open("wb") as dst:
        shutil.copyfileobj(src, dst)


def count_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and verify the bundled EndoCA manifests.")
    parser.add_argument("--suite", choices=["core", "diagnostic", "all"], default="all")
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--force", action="store_true", help="Overwrite existing extracted manifests.")
    args = parser.parse_args()

    selected = SUITES if args.suite == "all" else {args.suite: SUITES[args.suite]}
    report = {}
    for name, (archive_name, output_name, expected_rows) in selected.items():
        archive = args.manifest_dir / archive_name
        output = args.manifest_dir / output_name
        if not archive.exists():
            raise FileNotFoundError(f"Missing bundled manifest archive: {archive}")
        extract_manifest(archive, output, args.force)
        rows = count_rows(output)
        if rows != expected_rows:
            raise RuntimeError(f"{name} row count mismatch: expected {expected_rows}, got {rows}")
        report[name] = {"path": str(output), "rows": rows, "sha256": sha256(output)}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
