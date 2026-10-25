import copy

import pytest


def test_loso_numeric_order_fixed_members_and_deterministic_hash():
    from ppg_hr.data.splits import make_loso_splits

    result = make_loso_splits()
    assert result["subjects"] == [f"S{i}" for i in range(1, 16)]
    assert len(result["folds"]) == 15
    assert result["folds"][0]["test"] == ["S1"]
    assert result["folds"][0]["val"] == ["S2", "S3"]
    assert result["folds"][0]["train"] == [f"S{i}" for i in range(4, 16)]
    assert result["folds"][14]["test"] == ["S15"]
    assert result["folds"][14]["val"] == ["S1", "S2"]
    for fold in result["folds"]:
        assert tuple(len(fold[role]) for role in ("train", "val", "test")) == (12, 2, 1)
        assert set(fold["train"]) | set(fold["val"]) | set(fold["test"]) == set(result["subjects"])
    assert make_loso_splits(reversed(result["subjects"])) == result
    assert len(result["split_hash"]) == 64


@pytest.mark.parametrize("role", ["train", "val", "test"])
def test_subject_overlap_is_denied(role):
    from ppg_hr.data.splits import validate_subject_split
    from ppg_hr.errors import SplitLeakageError

    split = {"train": ["S1"], "val": ["S2"], "test": ["S3"]}
    other = "val" if role == "train" else "train"
    split[role].append(split[other][0])
    with pytest.raises(SplitLeakageError, match="overlap"):
        validate_subject_split(**split)


@pytest.mark.parametrize("subjects", [["S1"], [f"S{i}" for i in range(15)], ["S1"] * 15])
def test_loso_rejects_partial_wrong_or_duplicate_subject_inventory(subjects):
    from ppg_hr.data.splits import make_loso_splits
    from ppg_hr.errors import SplitLeakageError

    with pytest.raises(SplitLeakageError):
        make_loso_splits(subjects)


def test_split_completeness_and_duplicate_members_are_validated():
    from ppg_hr.data.splits import validate_subject_split
    from ppg_hr.errors import SplitLeakageError

    split = {"train": ["S1"], "val": ["S2"], "test": ["S3"]}
    validate_subject_split(**split, expected_subjects=["S1", "S2", "S3"])
    with pytest.raises(SplitLeakageError, match="inventory"):
        validate_subject_split(**split, expected_subjects=["S1", "S2", "S3", "S4"])
    duplicate = copy.deepcopy(split)
    duplicate["train"].append("S1")
    with pytest.raises(SplitLeakageError, match="duplicate"):
        validate_subject_split(**duplicate)
