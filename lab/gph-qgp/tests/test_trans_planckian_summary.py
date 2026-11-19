"""Tests for trans_planckian_summary (collapse sweep rows into composite verdicts)."""

import json
from pathlib import Path

import pandas as pd
import pytest

from gph_qgp import trans_planckian as tp
from gph_qgp import trans_planckian_summary as tps


def _study_df():
    return pd.DataFrame(tp.run_trans_planckian_study())


def test_collapse_channel_verdicts_shape():
    collapsed = tps.collapse_channel_verdicts(_study_df())
    assert len(collapsed) == 45  # one row per scale
    assert list(collapsed["n_constraints"].unique()) == [3]
    # sorted by scale ascending
    assert collapsed["scale_over_epl"].is_monotonic_increasing


def test_collapse_worst_case_logic():
    collapsed = tps.collapse_channel_verdicts(_study_df())
    first = collapsed.iloc[0]
    last = collapsed.iloc[-1]
    # lowest scale: every channel EXCLUDED -> strictest EXCLUDED
    assert first["strictest_verdict"] == "EXCLUDED"
    assert first["limiting_constraint"] == "Linear dispersion (n=1)"
    # highest scale: dispersion + birefringence SURVIVE, decay UNCONSTRAINED
    assert last["strictest_verdict"] == "SURVIVES"
    assert json.loads(last["channel_verdicts"]) == {
        "Linear dispersion (n=1)": "SURVIVES",
        "Birefringence (n=1)": "SURVIVES",
        "Superluminal decay (n=1)": "UNCONSTRAINED",
    }
    # explicit count columns add up to 3
    assert (
        int(last["n_excluded"]) + int(last["n_survives"]) + int(last["n_unconstrained"])
        == 3
    )
    assert int(last["n_survives"]) == 2
    assert int(last["n_unconstrained"]) == 1
    assert float(last["min_ratio"]) <= float(last["max_ratio"])


def test_collapse_empty_frame():
    assert tps.collapse_channel_verdicts(pd.DataFrame()).empty


def test_collapse_numeric_metric():
    df = pd.DataFrame(
        {
            "scale_over_epl": [1.0, 1.0, 10.0, 10.0],
            "log10_eqg_err": [0.1, 0.3, 0.2, 0.4],
        }
    )
    out = tps.collapse_numeric_metric(df, ["scale_over_epl"], "log10_eqg_err")
    assert len(out) == 2
    row = out[out["scale_over_epl"] == 1.0].iloc[0]
    assert row["log10_eqg_err_count"] == 2
    assert row["log10_eqg_err_mean"] == pytest.approx(0.2)
    assert row["log10_eqg_err_min"] == pytest.approx(0.1)
    assert row["log10_eqg_err_max"] == pytest.approx(0.3)


def test_collapse_numeric_metric_missing():
    assert tps.collapse_numeric_metric(pd.DataFrame(), ["scale_over_epl"], "x").empty
    assert tps.collapse_numeric_metric(
        pd.DataFrame({"a": [1]}), ["a"], "nope"
    ).empty


def test_export_study_bundle(tmp_path):
    rows = tp.run_trans_planckian_study()
    out = tps.export_study_bundle(rows, out_prefix=str(tmp_path / "bundle"))
    assert out["raw_rows"] == 135
    assert out["collapsed_rows"] == 45
    for key in ("raw_json", "raw_csv", "collapsed_json", "collapsed_csv"):
        assert Path(out[key]).exists()
    collapsed = pd.read_json(out["collapsed_json"])
    assert len(collapsed) == 45
    assert collapsed.iloc[0]["strictest_verdict"] == "EXCLUDED"
    assert collapsed.iloc[-1]["strictest_verdict"] == "SURVIVES"


def test_extract_verdict_transitions():
    raw = _study_df()
    collapsed = tps.collapse_channel_verdicts(raw)
    out = tps.extract_verdict_transitions(collapsed, raw)
    # strictest flips once: EXCLUDED -> SURVIVES, 11.10x -> 14.42x Planck
    # (matches the workbench's 45-point grid intervals)
    assert len(out["strictest_transitions"]) == 1
    tr = out["strictest_transitions"][0]
    assert tr["from_verdict"] == "EXCLUDED" and tr["to_verdict"] == "SURVIVES"
    assert tr["from_scale"] == pytest.approx(11.10, rel=1e-2)
    assert tr["to_scale"] == pytest.approx(14.42, rel=1e-2)
    assert tr["limiting_constraint"] == "Linear dispersion (n=1)"
    # four channel transitions: dispersion + birefringence to SURVIVES,
    # decay twice (EXCLUDED -> SURVIVES, then SURVIVES -> UNCONSTRAINED)
    assert len(out["channel_transitions"]) == 4
    by_channel: dict[str, list[dict]] = {}
    for c in out["channel_transitions"]:
        by_channel.setdefault(c["constraint"], []).append(c)
    assert set(by_channel) == {
        "Linear dispersion (n=1)",
        "Birefringence (n=1)",
        "Superluminal decay (n=1)",
    }
    assert by_channel["Linear dispersion (n=1)"][0]["to_verdict"] == "SURVIVES"
    assert by_channel["Birefringence (n=1)"][0]["to_verdict"] == "SURVIVES"
    decay = by_channel["Superluminal decay (n=1)"]
    assert [d["to_verdict"] for d in decay] == ["SURVIVES", "UNCONSTRAINED"]


def test_extract_verdict_transitions_empty():
    out = tps.extract_verdict_transitions(pd.DataFrame(), None)
    assert out == {"strictest_transitions": [], "channel_transitions": []}
