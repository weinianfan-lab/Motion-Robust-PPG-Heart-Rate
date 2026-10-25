"""Frozen LOSO with subject-level internal validation; never a window split."""

import hashlib
import json
from collections.abc import Iterable

from ppg_hr.errors import SplitLeakageError


def validate_subject_split(
    train: Iterable[str],
    val: Iterable[str],
    test: Iterable[str],
    *,
    expected_subjects: Iterable[str] | None = None,
) -> None:
    """Reject overlap, duplicate IDs, empty roles and incomplete inventories.

    This general guard also accepts small synthetic splits. The published
    ``loso_v1`` inventory and 12/2/1 sizes are enforced by ``make_loso_splits``.
    """
    groups = {"train": list(train), "val": list(val), "test": list(test)}
    for role, ids in groups.items():
        if not ids or any(not isinstance(subject, str) or not subject for subject in ids):
            raise SplitLeakageError(f"{role} must contain nonempty subject IDs")
        if len(ids) != len(set(ids)):
            raise SplitLeakageError(f"duplicate subject in {role}")
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = set(groups[first]) & set(groups[second])
        if overlap:
            raise SplitLeakageError(f"subject overlap between {first}/{second}: {sorted(overlap)}")
    if expected_subjects is not None:
        expected = list(expected_subjects)
        actual = set().union(*map(set, groups.values()))
        if len(expected) != len(set(expected)) or actual != set(expected):
            raise SplitLeakageError("split subject inventory differs from expected inventory")


def make_loso_splits(subjects: Iterable[str] | None = None) -> dict:
    """Return the 15-subject protocol and its canonical SHA-256 identity.

    The hash is SHA-256 of UTF-8, sorted-key, compact JSON of the returned
    object *without* its ``split_hash`` field. Input ordering cannot change it.
    """
    expected = [f"S{i}" for i in range(1, 16)]
    provided = expected if subjects is None else list(subjects)
    if len(provided) != 15 or set(provided) != set(expected):
        raise SplitLeakageError("loso_v1 requires exactly the S1 through S15 subject inventory")
    folds = []
    for index, subject in enumerate(expected):
        val = [expected[(index + 1) % 15], expected[(index + 2) % 15]]
        train = [item for item in expected if item not in [subject, *val]]
        validate_subject_split(train, val, [subject], expected_subjects=expected)
        folds.append({"fold": index, "train": train, "val": val, "test": [subject]})
    result = {
        "protocol_id": "loso_v1",
        "description": "LOSO + fixed subject-level internal validation; no final 14-subject refit",
        "subjects": expected,
        "folds": folds,
    }
    payload = json.dumps(result, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {**result, "split_hash": hashlib.sha256(payload).hexdigest()}
