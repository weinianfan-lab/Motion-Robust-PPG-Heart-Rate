"""Local ZIP integrity checks; hashes do not authenticate the transport or source.

The caller must keep the archive unchanged while inspecting/extracting it and bind
these receipts to independently documented official transport evidence. This
module never downloads files, deserializes data, or overwrites an extraction.
"""

import hashlib
import os
import shutil
import stat
import zipfile
from pathlib import Path, PureWindowsPath

from ppg_hr.errors import DataValidationError
from ppg_hr.io import write_json

_STATUS_FILE = ".acquisition-status.json"
_CHUNK_BYTES = 1024 * 1024


def _validated_members(archive: zipfile.ZipFile) -> list[tuple[zipfile.ZipInfo, str]]:
    members = []
    seen = {}
    for info in archive.infolist():
        name = info.orig_filename.replace("\\", "/")
        parts = [part for part in name.split("/") if part not in ("", ".")]
        if (
            not parts
            or name.startswith("/")
            or PureWindowsPath(name).drive
            or ".." in parts
            or any(
                part.endswith((".", " "))
                or any(ord(char) < 32 or char in ':<>"|?*' for char in part)
                or PureWindowsPath(part).is_reserved()
                for part in parts
            )
            or parts[0].casefold() == _STATUS_FILE.casefold()
        ):
            raise DataValidationError("Archive contains an unsafe or reserved member path")
        file_type = stat.S_IFMT(info.external_attr >> 16)
        if file_type not in (0, stat.S_IFREG, stat.S_IFDIR):
            raise DataValidationError("Archive contains a symlink or unsupported special file")
        if info.is_dir() and info.file_size != 0:
            raise DataValidationError("Archive directory contains unexpected payload bytes")
        normalized = "/".join(parts)
        key = normalized.casefold()
        if key in seen:
            raise DataValidationError("Archive contains duplicate normalized member paths")
        seen[key] = info.is_dir()
        members.append((info, normalized))
    for key in seen:
        parts = key.split("/")
        if any(seen.get("/".join(parts[:index])) is False for index in range(1, len(parts))):
            raise DataValidationError("Archive member is both a file and a parent directory")
    return sorted(members, key=lambda member: member[1])


def inspect_archive(
    path: Path,
    expected_bytes: int | None = None,
    expected_sha1: str | None = None,
    expected_md5: str | None = None,
) -> dict:
    """Hash all archive bytes and inspect paths/metadata; CRC awaits extraction."""
    path = Path(path)
    try:
        with path.open("rb") as source:
            initial = os.fstat(source.fileno())
            if expected_bytes is not None and initial.st_size != expected_bytes:
                raise DataValidationError("Archive byte count does not match the expected size")
            with zipfile.ZipFile(source) as archive:
                members = _validated_members(archive)
            source.seek(0)
            hashes = {name: hashlib.new(name) for name in ("sha256", "sha1", "md5")}
            byte_count = 0
            while chunk := source.read(_CHUNK_BYTES):
                byte_count += len(chunk)
                for digest in hashes.values():
                    digest.update(chunk)
            final = os.fstat(source.fileno())
            if (initial.st_size, initial.st_mtime_ns) != (final.st_size, final.st_mtime_ns):
                raise DataValidationError("Archive changed during inspection")
            if byte_count != initial.st_size:
                raise DataValidationError("Archive read byte count does not match its size")
    except (OSError, zipfile.BadZipFile) as error:
        raise DataValidationError("Cannot inspect archive bytes or ZIP metadata") from error
    digests = {name: digest.hexdigest() for name, digest in hashes.items()}
    for name, expected in (("sha1", expected_sha1), ("md5", expected_md5)):
        if expected is not None and digests[name] != expected.strip().lower():
            raise DataValidationError(f"Archive {name} does not match the expected digest")
    inventory = [
        {
            "path": name,
            "is_directory": info.is_dir(),
            "bytes": info.file_size,
            "compressed_bytes": info.compress_size,
            "crc32": f"{info.CRC:08x}",
        }
        for info, name in members
    ]
    directory_count = sum(member["is_directory"] for member in inventory)
    return {
        "schema_version": 1,
        "status": "inspected",
        "bytes": byte_count,
        **digests,
        "member_count": len(inventory),
        "file_count": len(inventory) - directory_count,
        "directory_count": directory_count,
        "uncompressed_bytes": sum(member["bytes"] for member in inventory),
        "members": inventory,
        "crc_verified": False,
    }


def extract_archive(path: Path, destination: Path) -> dict:
    """CRC-check every file while copying into a new directory, preserving failures.

    Only a receipt with status ``ready`` denotes a completed extraction. A process
    interruption may leave ``incomplete``; handled failures leave ``failed``. Such
    directories are retained as evidence and cannot be reused by this function.
    """
    path, destination = Path(path), Path(destination)
    if destination.exists() or destination.is_symlink():
        raise DataValidationError("Extraction destination must be a new directory")
    receipt = inspect_archive(path)
    initial = path.stat()
    required_bytes = receipt["uncompressed_bytes"] + max(
        64 * _CHUNK_BYTES, receipt["uncompressed_bytes"] // 100
    )
    if shutil.disk_usage(destination.parent).free < required_bytes:
        raise DataValidationError("Insufficient free disk space for extraction plus safety reserve")
    try:
        destination.mkdir(exist_ok=False)
    except OSError as error:
        raise DataValidationError("Cannot create a new extraction directory") from error
    receipt.update(status="incomplete", files=[])
    status_path = destination / _STATUS_FILE
    root = destination.resolve()
    write_json(status_path, receipt)
    try:
        with zipfile.ZipFile(path) as archive:
            members = _validated_members(archive)
            for info, name in members:
                target = destination.joinpath(*name.split("/"))
                if not target.resolve().is_relative_to(root):
                    raise DataValidationError("Extraction target escapes the new directory")
                if info.is_dir():
                    with archive.open(info) as source:
                        source.read()  # Even an empty directory must satisfy its stored CRC.
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                byte_count = 0
                with archive.open(info) as source, target.open("xb") as output:
                    while chunk := source.read(_CHUNK_BYTES):
                        byte_count += len(chunk)
                        if byte_count > info.file_size:
                            raise DataValidationError("Extracted bytes exceed archive member size")
                        output.write(chunk)
                        digest.update(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                if byte_count != info.file_size:
                    raise DataValidationError("Extracted byte count does not match archive member")
                receipt["files"].append(
                    {"path": name, "bytes": byte_count, "sha256": digest.hexdigest()}
                )
        final = path.stat()
        if (initial.st_size, initial.st_mtime_ns) != (final.st_size, final.st_mtime_ns):
            raise DataValidationError("Archive changed during extraction")
        receipt.update(status="ready", crc_verified=True)
        write_json(status_path, receipt)
        return receipt
    except Exception as error:
        receipt.update(status="failed", crc_verified=False, failure_type=type(error).__name__)
        write_json(status_path, receipt)
        raise DataValidationError(
            "Archive extraction failed; partial evidence has been retained"
        ) from error
