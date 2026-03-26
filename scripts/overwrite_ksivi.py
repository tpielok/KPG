#!/usr/bin/env python3
# Reference upstream repository: https://github.com/longinyu/ksivi
# This file is part of the KPG overlay package.

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Overwrite a local KSIVI checkout with release files.')
    parser.add_argument('--ksivi-dir', required=True, help='Path to local KSIVI checkout')
    parser.add_argument('--dry-run', action='store_true', help='Print actions without copying')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    ksivi_dir = Path(args.ksivi_dir).resolve()

    if not ksivi_dir.exists():
        raise SystemExit(f'KSIVI directory does not exist: {ksivi_dir}')
    if not (ksivi_dir / '.git').exists():
        raise SystemExit(f'Target is not a git checkout: {ksivi_dir}')

    skip_prefixes = {'scripts/'}
    skip_files = {'README.md'}

    to_copy = []
    for src in repo_root.rglob('*'):
        if not src.is_file():
            continue
        rel = src.relative_to(repo_root).as_posix()
        if rel in skip_files:
            continue
        if any(rel.startswith(prefix) for prefix in skip_prefixes):
            continue
        to_copy.append((src, ksivi_dir / rel))

    print(f'Preparing to copy {len(to_copy)} file(s) into: {ksivi_dir}')
    for src, dst in sorted(to_copy, key=lambda t: t[0].as_posix()):
        print(f'{src.relative_to(repo_root).as_posix()} -> {dst}')
        if args.dry_run:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    if args.dry_run:
        print('Dry run completed; no files copied.')
    else:
        print('Overwrite completed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
