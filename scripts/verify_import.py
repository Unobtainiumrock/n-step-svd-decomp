"""Verify the handoff snapshot; optionally verify the original working-tree bytes."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Original companion directory, if available")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "docs/import-manifest.json").read_text())
    failures = []
    for entry in manifest["files"]:
        target = root / entry["destination"]
        if not target.is_file() or digest(target) != entry["handoff_sha256"]:
            failures.append("Handoff mismatch: " + entry["destination"])
        if args.source:
            source = args.source / entry["source"]
            if not source.is_file() or digest(source) != entry["source_sha256"]:
                failures.append("Source mismatch: " + entry["source"])
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Verified {len(manifest['files'])} imported files" + (" and original source bytes" if args.source else ""))


if __name__ == "__main__":
    main()
