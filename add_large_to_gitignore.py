#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Scan current directory for files larger than N MB (default 100 MB)
and append their repo-root-relative paths to .gitignore if not present.
Skips the .git directory. Works on Windows/macOS/Linux.
"""

from __future__ import annotations
import os
import argparse
from pathlib import Path
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Add large files to .gitignore")
    parser.add_argument("--mb", type=float, default=100.0,
                        help="Size threshold in MB (default: 100)")
    args = parser.parse_args()

    root = Path(".").resolve()
    threshold_bytes = int(args.mb * 1024 * 1024)
    git_dir = root / ".git"
    gi_path = root / ".gitignore"

    # Load existing .gitignore entries (trim comments/whitespace)
    existing: set[str] = set()
    if gi_path.exists():
        for line in gi_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            existing.add(s)

    to_add: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip .git directory
        if Path(dirpath) == git_dir:
            dirnames[:] = []  # don't descend
            continue
        # Also avoid descending into any .git subdirs just in case
        dirnames[:] = [d for d in dirnames if not (Path(dirpath) / d) == git_dir]

        for fn in filenames:
            p = Path(dirpath) / fn
            try:
                size = p.stat().st_size
            except OSError:
                continue  # skip unreadable files
            if size >= threshold_bytes:
                rel = p.relative_to(root).as_posix()  # forward slashes for .gitignore
                pattern = "/" + rel  # anchor to repo root
                if pattern not in existing:
                    to_add.append(pattern)

    if not to_add:
        print(f"No files >= {args.mb} MB found.")
        return

    header = f"# auto-added large files ({args.mb:.2f} MB+) on {datetime.now():%Y-%m-%d %H:%M:%S}"
    with gi_path.open("a", encoding="utf-8") as f:
        f.write("\n" + header + "\n")
        for item in to_add:
            f.write(item + "\n")

    print(f"Added {len(to_add)} path(s) to {gi_path.name}:")
    for item in to_add:
        print("  " + item)

if __name__ == "__main__":
    main()
