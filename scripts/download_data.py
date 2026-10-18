"""Download every raw dataset file used by the project, pinned to exact revisions, and verify checksums.

Usage: python -P scripts/download_data.py
Reads data/raw_manifest.json (path, url, sha256 for each file) and writes files under data/raw/.
Files already present with the right checksum are skipped. Then run scripts/prepare_data.py and
scripts/prepare_ood_data.py to rebuild data/processed/.
"""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "raw_manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    entries = json.loads(MANIFEST.read_text())["files"]
    bad = []
    for e in entries:
        dest = ROOT / e["path"]
        if dest.exists() and sha256(dest) == e["sha256"]:
            print(f"ok      {e['path']}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"fetch   {e['path']}", flush=True)
        urllib.request.urlretrieve(e["url"], dest)
        if sha256(dest) != e["sha256"]:
            bad.append(e["path"])
            print(f"MISMATCH {e['path']}")
    if bad:
        sys.exit(f"{len(bad)} file(s) failed checksum: {bad}")
    print(f"all {len(entries)} files verified")


if __name__ == "__main__":
    main()
