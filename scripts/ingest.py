"""Walk data/raw/, normalize each file via the loader, write JSON to data/processed/.

Usage:
    python -m scripts.ingest                                # uses defaults
    python -m scripts.ingest --raw mydocs --processed out
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.config import settings
from src.loaders import UnsupportedFileTypeError, get_loader


def ingest_directory(raw_dir: Path, processed_dir: Path) -> tuple[int, int, int]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    ok = skipped = failed = 0

    files = sorted(
        p for p in raw_dir.rglob("*") if p.is_file() and p.name != ".gitkeep"
    )
    if not files:
        print(f"No files found in {raw_dir}")
        return (0, 0, 0)

    for file_path in files:
        try:
            loader = get_loader(file_path)
        except UnsupportedFileTypeError as exc:
            print(f"SKIP {file_path.name}: {exc}")
            skipped += 1
            continue

        try:
            documents = loader.load(file_path)
        except Exception as exc:  # noqa: BLE001 — boundary
            print(
                f"FAIL {file_path.name}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            failed += 1
            continue

        relative = file_path.relative_to(raw_dir)
        output_path = processed_dir / relative.with_suffix(relative.suffix + ".json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [doc.to_dict() for doc in documents]
        output_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"OK   {file_path.name} -> {output_path.relative_to(processed_dir.parent)}")
        ok += 1

    return (ok, skipped, failed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize raw docs into JSON.")
    parser.add_argument("--raw", type=Path, default=settings.raw_dir)
    parser.add_argument("--processed", type=Path, default=settings.processed_dir)
    args = parser.parse_args()

    if not args.raw.exists():
        print(f"Raw directory does not exist: {args.raw}", file=sys.stderr)
        sys.exit(1)

    ok, skipped, failed = ingest_directory(args.raw, args.processed)
    print(f"\nDone. ok={ok} skipped={skipped} failed={failed}")
    if failed:
        sys.exit(2)


if __name__ == "__main__":
    main()
