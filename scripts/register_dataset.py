"""Validate the pinned author-provided ZIP and extract to a new external directory."""

import argparse
import json
import sys
from pathlib import Path

from ppg_hr.data.acquisition import extract_archive, inspect_archive
from ppg_hr.errors import PPGError

ARCHIVE_SHA256 = "fcda4d13f6475e88a24f1ef9658627dc98e04dd5c5dd790cd2b0a1a22e31913b"
ARCHIVE_BYTES = 2864337783


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--trust-official-archive", action="store_true", required=True)
    args = parser.parse_args()
    try:
        identity = inspect_archive(args.archive, expected_bytes=ARCHIVE_BYTES)
        if identity["sha256"] != ARCHIVE_SHA256:
            raise ValueError("ZIP differs from the pinned author release; a new audit is required")
        project = Path(__file__).resolve().parents[1]
        if args.archive.resolve().is_relative_to(
            project
        ) or args.destination.resolve().is_relative_to(project):
            raise ValueError("raw archive and extraction must stay outside the repository")
        receipt = extract_archive(args.archive, args.destination)
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "data_root": str(args.destination),
                    "files": len(receipt["files"]),
                    "crc_verified": receipt["crc_verified"],
                }
            )
        )
    except (PPGError, ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
