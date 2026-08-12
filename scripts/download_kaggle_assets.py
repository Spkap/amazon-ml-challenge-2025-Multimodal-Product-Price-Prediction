#!/usr/bin/env python3
"""Download a Kaggle dataset only after checking its declared size.

Authentication is delegated to the official Kaggle package. This script never
reads, prints or writes credential values itself.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

DEFAULT_MAX_BYTES = 100 * 1024 * 1024
DISK_HEADROOM_BYTES = 256 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect an authenticated Kaggle dataset and download only when "
            "every selected file and the aggregate selection fit the size cap."
        )
    )
    parser.add_argument("dataset", help="Kaggle dataset reference: owner/slug")
    parser.add_argument(
        "--file",
        action="append",
        dest="files",
        help="Download only this exact dataset path; repeat to select multiple files.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("data/raw/kaggle"),
        help="Destination root (default: data/raw/kaggle)",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=DEFAULT_MAX_BYTES,
        help=f"Maximum file and aggregate bytes (default: {DEFAULT_MAX_BYTES})",
    )
    parser.add_argument(
        "--allow-large",
        action="store_true",
        help="Explicitly bypass the byte cap; free-disk checks still apply.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Print the authenticated manifest without downloading.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow the Kaggle client to replace an existing download.",
    )
    args = parser.parse_args()
    if args.max_bytes <= 0:
        parser.error("--max-bytes must be a positive integer")
    if args.dataset.count("/") != 1 or any(
        not part.strip() for part in args.dataset.split("/")
    ):
        parser.error("dataset must have the form owner/slug")
    return args


def value(obj: Any, *names: str) -> Any:
    """Read the first populated attribute or key used by Kaggle API versions."""
    for name in names:
        candidate = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
        if candidate is not None:
            return candidate
    return None


def file_record(item: Any) -> tuple[str, int]:
    name = value(item, "name", "file_name", "fileName")
    size = value(item, "total_bytes", "totalBytes", "size")
    if not isinstance(name, str) or size is None:
        raise RuntimeError(f"Unsupported Kaggle file metadata: {item!r}")
    return name, int(size)


def choose_files(
    records: Iterable[tuple[str, int]], requested: list[str] | None
) -> list[tuple[str, int]]:
    available = dict(records)
    if not requested:
        return sorted(available.items())
    missing = sorted(set(requested) - available.keys())
    if missing:
        formatted = "\n  - ".join(missing)
        raise SystemExit(f"Requested file(s) not present:\n  - {formatted}")
    return [(name, available[name]) for name in requested]


def human_bytes(size: int) -> str:
    value_float = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value_float < 1024 or unit == "TiB":
            return f"{value_float:.2f} {unit}"
        value_float /= 1024
    raise AssertionError("unreachable")


def enforce_limits(
    selection: list[tuple[str, int]],
    max_bytes: int,
    allow_large: bool,
    destination: Path,
) -> int:
    total = sum(size for _, size in selection)
    oversized = [(name, size) for name, size in selection if size > max_bytes]
    if not allow_large and (oversized or total > max_bytes):
        details = "\n".join(
            f"  - {name}: {size} bytes ({human_bytes(size)})"
            for name, size in oversized
        )
        if details:
            details = f"\nFiles over the cap:\n{details}"
        raise SystemExit(
            "Download refused by size policy. "
            f"Selected {total} bytes ({human_bytes(total)}); "
            f"cap is {max_bytes} bytes ({human_bytes(max_bytes)})."
            f"{details}\nUse --file for a smaller selection, increase --max-bytes, "
            "or pass --allow-large after checking disk space."
        )

    destination.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(destination).free
    required = total + DISK_HEADROOM_BYTES
    if required > free_bytes:
        raise SystemExit(
            "Download refused: insufficient free disk after reserving "
            f"{human_bytes(DISK_HEADROOM_BYTES)} headroom. "
            f"Required {human_bytes(required)}, available {human_bytes(free_bytes)}."
        )
    return total


def main() -> int:
    args = parse_args()
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print(
            "The official Kaggle client is required. Install the project's "
            "optional Kaggle dependency before running this script.",
            file=sys.stderr,
        )
        return 2

    api = KaggleApi()
    api.authenticate()
    response = api.dataset_list_files(args.dataset)
    raw_files = value(response, "files") or []
    all_records = [file_record(item) for item in raw_files]
    if not all_records:
        raise SystemExit(
            "Kaggle returned no file metadata; refusing to issue an unchecked download."
        )
    selection = choose_files(all_records, args.files)

    print(f"Kaggle dataset: {args.dataset}")
    for name, size in selection:
        print(f"  {size:>12} bytes  {name}")
    total_selected = sum(size for _, size in selection)
    print(f"Selected total: {total_selected} bytes ({human_bytes(total_selected)})")
    if args.list_only:
        return 0

    owner, slug = args.dataset.split("/", maxsplit=1)
    destination = args.dest.resolve() / owner / slug
    total = enforce_limits(
        selection, args.max_bytes, args.allow_large, destination.parent
    )
    print(f"Approved download: {human_bytes(total)} -> {destination}")

    if args.files:
        destination.mkdir(parents=True, exist_ok=True)
        for name, _ in selection:
            api.dataset_download_file(
                args.dataset,
                name,
                path=str(destination),
                force=args.force,
                quiet=False,
            )
    else:
        api.dataset_download_files(
            args.dataset,
            path=str(destination),
            force=args.force,
            quiet=False,
            unzip=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
