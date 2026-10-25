"""Channel timing fixtures are generated numeric data, never participant records."""

import json
import pickle

import numpy as np

from ppg_hr.data.adapter import SOURCE_RULES
from ppg_hr.data.audit import audit_dataset


def synthetic_payload():
    return {
        "subject": "S1",
        "signal": {
            "wrist": {
                "BVP": np.arange(768, dtype=float)[:, None],
                "ACC": np.ones((384, 3)),
                "EDA": np.full((44, 1), 123.0),
                "TEMP": np.full((40, 1), 456.0),
            },
            "chest": {
                "ECG": np.zeros((7000, 1)),
                "ACC": np.zeros((3500, 3)),
                "Resp": np.zeros((2100, 1)),
                "EDA": np.zeros((1400, 1)),
                "EMG": np.zeros((700, 1)),
                "Temp": np.zeros((350, 1)),
            },
        },
        "activity": np.ones(48, dtype=int),
        "label": np.array([60.0, 75.0, 90.0]),
        "rpeaks": np.array([1.0, 2.1, 3.2]),
    }


def audit_payload(tmp_path, payload):
    (tmp_path / "S1.pkl").write_bytes(pickle.dumps(payload))
    return audit_dataset(tmp_path, trust_local_pickle=True)["subjects"][0]


def test_every_documented_channel_duration_uses_its_first_axis_and_own_rate(tmp_path):
    report = audit_payload(tmp_path, synthetic_payload())
    expected = {
        "signal.wrist.BVP": (64, 12.0),
        "signal.wrist.ACC": (32, 12.0),
        "signal.wrist.EDA": (4, 11.0),
        "signal.wrist.TEMP": (4, 10.0),
        "signal.chest.ECG": (700, 10.0),
        "signal.chest.ACC": (700, 5.0),
        "signal.chest.Resp": (700, 3.0),
        "signal.chest.EDA": (700, 2.0),
        "signal.chest.EMG": (700, 1.0),
        "signal.chest.Temp": (700, 0.5),
        "rpeaks": (None, None),
    }
    assert report["documented_channel_sample_rates_hz"] == {
        name: values[0] for name, values in expected.items()
    }
    assert report["signal_duration_s"] == {name: values[1] for name, values in expected.items()}
    for name, (rate, duration) in expected.items():
        signal = report["signals"][name]
        assert signal["documented_sample_rate_hz"] == rate
        assert signal["duration_s"] == duration
        assert signal["sample_rate_basis"]
        assert signal["duration_basis"]
    assert report["duration_s"] == {"bvp": 12.0, "acc": 12.0, "activity": 12.0, "ecg": 10.0}
    assert report["window_counts"] == {"bvp": 3, "acc": 3, "labels": 3}
    assert report["manifest_summary"]["windows"] == 3
    assert report["status"] == "BLOCKED"
    assert all(blocker in report["blockers"] for blocker in SOURCE_RULES["blockers"])
    json.dumps(report, allow_nan=False)


def test_unknown_channels_and_events_have_no_invented_rate_or_duration(tmp_path):
    payload = synthetic_payload()
    payload["signal"]["wrist"]["UNKNOWN"] = np.zeros((400, 2))
    payload["signal"]["chest"]["UNKNOWN"] = np.zeros((700, 1))
    payload["signal"]["other"] = {"BVP": np.zeros((640, 1))}
    report = audit_payload(tmp_path, payload)
    for name in ("signal.wrist.UNKNOWN", "signal.chest.UNKNOWN", "signal.other.BVP", "rpeaks"):
        signal = report["signals"][name]
        assert report["documented_channel_sample_rates_hz"][name] is None
        assert report["signal_duration_s"][name] is None
        assert signal["documented_sample_rate_hz"] is None
        assert signal["duration_s"] is None
        assert signal["units"] == "unknown"
        assert signal["sample_rate_basis"]
        assert signal["duration_basis"]
    assert report["signals"]["rpeaks"]["signal_kind"] == "event_vector"
    assert "uniform" in report["signals"]["rpeaks"]["sample_rate_basis"]


def test_documentary_raw_csv_units_do_not_assert_pickle_scaling_or_rescale_values(tmp_path):
    report = audit_payload(tmp_path, synthetic_payload())
    expected_raw_units = {"ACC": "1/64 g", "EDA": "uS", "TEMP": "degrees C"}
    for name, documented_unit in expected_raw_units.items():
        signal = report["signals"][f"signal.wrist.{name}"]
        assert signal["units"] == "unknown"
        assert signal["documented_raw_unit"] == documented_unit
        assert "not independently verified" in signal["units_basis"]
    assert report["signals"]["signal.wrist.EDA"]["mean"] == 123.0
    assert report["signals"]["signal.wrist.TEMP"]["mean"] == 456.0
    assert report["signals"]["signal.chest.ECG"]["units"] == "unknown"


def test_scalar_optional_channel_has_no_fabricated_time_axis(tmp_path):
    payload = synthetic_payload()
    payload["signal"]["wrist"]["EDA"] = np.array(1.0)
    del payload["signal"]["chest"]["ECG"]
    report = audit_payload(tmp_path, payload)
    signal = report["signals"]["signal.wrist.EDA"]
    assert signal["shape"] == []
    assert signal["documented_sample_rate_hz"] == 4
    assert signal["duration_s"] is None
    assert "sample axis" in signal["duration_basis"]
    assert report["duration_s"]["ecg"] is None
