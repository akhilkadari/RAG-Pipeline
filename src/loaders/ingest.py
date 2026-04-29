from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from src.loaders import UnsupportedFileTypeError, get_loader

def ingest_directory (raw_dir: Path, processed_dir: Path) -> None:
    # creates the processed directory 
    processed_dir.mkdir(parents=True, exist_ok=True)
    ok = skipped = failed = 0

    # recursively finds all files in the raw directory and sorts them
    files = sorted(p for p in raw_dir.rglob("*") if p.is_file() and p.name != ".gitkeep")

    # if no files are found, print a message and return the counts
    if not files:
        print(f"No files found in {raw_dir}")
        return (0, 0, 0)
    
    # loops through each file in "files" list
    for file_path in files:
        # picks a loader with get_loader
        try:
            loader = get_loader(file_path)
        except UnsupportedFileTypeError as exc:
            # if file is not supported, skip it and print a message
            print(f"SKIP {file_path.name}: {exc}")
            skipped += 1
            continue
        
        # gets the documents from the loader
        try:
            documents = loader.load(file_path)
        except Exception as exc:
            # if an error occurs, print a message and increment the failed count
            print(f"FAIL {file_path.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            failed += 1
            continue

    # gets the relative path of the file and creates the output path
    relative = file_path.relative_to(raw_dir)
    # creates the output path with the same extension but .json
    output_path = processed_dir / relative.with_suffix(relative.suffix + ".json")
    # creates the parent directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # converts the documents to a list of dictionaries
    payload = [doc.to_dict() for doc in documents]
    # writes the payload to the output path
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"OK   {file_path.name} -> {output_path.relative_to(processed_dir.parent)}")
    ok += 1

    return (ok, skipped, failed)

