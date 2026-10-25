import hashlib
import json
import stat
import warnings
import zipfile
from types import SimpleNamespace

import pytest

from ppg_hr.data.acquisition import extract_archive, inspect_archive
from ppg_hr.errors import DataValidationError


def make_archive(tmp_path, members):
    path = tmp_path / "tiny.zip"
    with (
        warnings.catch_warnings(),
        zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive,
    ):
        warnings.filterwarnings("ignore", message="Duplicate name:", category=UserWarning)
        for name, content in members:
            archive.writestr(name, content)
    return path


def test_inspect_reports_byte_identity_and_full_inventory(tmp_path):
    path = make_archive(tmp_path, [("S1/", b""), ("S1/a.txt", b"abc"), ("empty", b"")])
    raw = path.read_bytes()
    receipt = inspect_archive(
        path,
        expected_bytes=len(raw),
        expected_sha1=hashlib.sha1(raw).hexdigest().upper(),
        expected_md5=hashlib.md5(raw).hexdigest(),
    )
    assert receipt["bytes"] == len(raw)
    for algorithm in ("sha256", "sha1", "md5"):
        assert receipt[algorithm] == hashlib.new(algorithm, raw).hexdigest()
    assert receipt["member_count"] == 3
    assert receipt["file_count"] == 2
    assert receipt["directory_count"] == 1
    assert receipt["uncompressed_bytes"] == 3
    assert {member["path"] for member in receipt["members"]} == {"S1", "S1/a.txt", "empty"}
    assert receipt["crc_verified"] is False


@pytest.mark.parametrize(
    "expectation",
    [
        {"expected_bytes": 1},
        {"expected_sha1": "0" * 40},
        {"expected_md5": "0" * 32},
    ],
)
def test_inspect_rejects_byte_count_or_expected_digest_mismatch(tmp_path, expectation):
    path = make_archive(tmp_path, [("a.txt", b"abc")])
    original = path.read_bytes()
    with pytest.raises(DataValidationError):
        inspect_archive(path, **expectation)
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "name",
    [
        "../escape",
        "a/../../escape",
        "/absolute",
        "\\absolute",
        "C:/escape",
        "C:escape",
        "a\\..\\escape",
        "//host/share/file",
        "a:stream",
        "trailing. /x",
        "CON/x",
        ".acquisition-status.json",
    ],
)
def test_unsafe_members_never_create_destination(tmp_path, name):
    path = make_archive(tmp_path, [(name, b"abc")])
    destination = tmp_path / "new"
    with pytest.raises(DataValidationError):
        extract_archive(path, destination)
    assert not destination.exists()
    assert not (tmp_path / "escape").exists()


@pytest.mark.parametrize(
    "names",
    [
        ("a/b", "a\\b"),
        ("a/b", "A/B"),
        ("a/b", "a/./b"),
        ("a/b", "a//b"),
        ("a", "a/b"),
        ("A", "a/b"),
        ("a/", "A/"),
    ],
)
def test_duplicate_or_file_directory_aliases_are_rejected_before_writes(tmp_path, names):
    path = make_archive(tmp_path, [(name, b"") for name in names])
    destination = tmp_path / "new"
    with pytest.raises(DataValidationError):
        extract_archive(path, destination)
    assert not destination.exists()


def test_zip_symlink_is_rejected_before_writes(tmp_path):
    member = zipfile.ZipInfo("link")
    member.create_system = 3
    member.external_attr = (stat.S_IFLNK | 0o777) << 16
    path = make_archive(tmp_path, [(member, b"../outside")])
    with pytest.raises(DataValidationError):
        extract_archive(path, tmp_path / "new")
    assert not (tmp_path / "new").exists()


def test_extraction_preserves_existing_destination(tmp_path):
    path = make_archive(tmp_path, [("a.txt", b"abc")])
    destination = tmp_path / "existing"
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_bytes(b"untouched")
    with pytest.raises(DataValidationError):
        extract_archive(path, destination)
    assert list(destination.iterdir()) == [sentinel]
    assert sentinel.read_bytes() == b"untouched"


def test_valid_extractions_are_identical_crc_verified_and_hash_every_file(tmp_path):
    path = make_archive(tmp_path, [("S1/a.txt", b"abc"), ("empty", b"")])
    first = extract_archive(path, tmp_path / "first")
    second = extract_archive(path, tmp_path / "second")
    assert first == second
    assert first["status"] == "ready"
    assert first["crc_verified"] is True
    assert first["files"] == [
        {
            "path": "S1/a.txt",
            "bytes": 3,
            "sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        },
        {
            "path": "empty",
            "bytes": 0,
            "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        },
    ]
    assert (tmp_path / "first" / "S1" / "a.txt").read_bytes() == b"abc"
    assert json.loads((tmp_path / "first" / ".acquisition-status.json").read_text()) == first
    assert str(tmp_path) not in json.dumps(first)


def test_bad_crc_leaves_failed_evidence_and_no_ready_status(tmp_path):
    path = make_archive(tmp_path, [("a-good.txt", b"good"), ("z-bad.txt", b"abc")])
    path.write_bytes(path.read_bytes().replace(b"abc", b"xbc", 1))
    destination = tmp_path / "failed"
    with pytest.raises(DataValidationError):
        extract_archive(path, destination)
    assert (destination / "a-good.txt").read_bytes() == b"good"
    receipt = json.loads((destination / ".acquisition-status.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["crc_verified"] is False
    assert [item["path"] for item in receipt["files"]] == ["a-good.txt"]
    with pytest.raises(DataValidationError):
        extract_archive(path, destination)


def test_insufficient_disk_space_does_not_start_extraction(tmp_path, monkeypatch):
    import shutil

    path = make_archive(tmp_path, [("a.txt", b"abc")])
    monkeypatch.setattr(shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    with pytest.raises(DataValidationError):
        extract_archive(path, tmp_path / "new")
    assert not (tmp_path / "new").exists()


def test_directory_crc_is_checked_before_marking_archive_ready(tmp_path):
    path = make_archive(tmp_path, [("empty-directory/", b"")])
    corrupted = bytearray(path.read_bytes())
    crc_offset = corrupted.index(b"PK\x01\x02") + 16
    corrupted[crc_offset : crc_offset + 4] = (1).to_bytes(4, "little")
    path.write_bytes(corrupted)
    destination = tmp_path / "failed"
    with pytest.raises(DataValidationError):
        extract_archive(path, destination)
    receipt = json.loads((destination / ".acquisition-status.json").read_text())
    assert receipt["status"] == "failed"
    assert receipt["crc_verified"] is False
