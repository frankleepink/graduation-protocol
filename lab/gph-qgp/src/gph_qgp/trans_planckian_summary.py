"""Trans-Planckian study summary: collapse sweep rows into composite verdicts.

"Collapse" here means data aggregation, not wavefunction collapse: it
takes the per-constraint rows from ``trans_planckian.run_trans_planckian_study``
and rolls them up to one row per scale, using worst-case (highest-severity)
verdict logic -- EXCLUDED beats SURVIVES beats UNCONSTRAINED -- and the
highest bound ratio as the limiting constraint. Also aggregates numeric
fit metrics (e.g. log10 E_QG error) across runs, and exports a complete
raw + collapsed study bundle as CSV and JSON.

Requires pandas (added to the package dependencies with this module).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

VERDICT_SEVERITY: dict[str, int] = {
    "EXCLUDED": 3,
    "SURVIVES": 2,
    "UNCONSTRAINED": 1,
}
SEVERITY_TO_VERDICT: dict[int, str] = {v: k for k, v in VERDICT_SEVERITY.items()}


def collapse_channel_verdicts(
    df: pd.DataFrame,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Collapse multiple constraint rows per group into worst-case verdicts and ratios."""
    if group_cols is None:
        group_cols = [c for c in ["scale_over_epl", "eff_scale_gev"] if c in df.columns]
    if df.empty or not group_cols:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for keys, grp in df.groupby(group_cols, dropna=False):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        row: dict[str, Any] = dict(zip(group_cols, key_tuple))

        grp_sorted = grp.sort_values("ratio", ascending=False)
        top = grp_sorted.iloc[0]

        severities = [VERDICT_SEVERITY.get(str(v), 0) for v in grp["verdict"]]
        strictest_verdict = SEVERITY_TO_VERDICT.get(max(severities), "UNKNOWN")
        counts = pd.Series(grp["verdict"]).value_counts().to_dict()
        verdict_map = {str(r["constraint"]): str(r["verdict"]) for _, r in grp.iterrows()}

        row.update({
            "n_constraints": int(len(grp)),
            "strictest_verdict": strictest_verdict,
            "max_ratio": float(top["ratio"]),
            "min_ratio": float(grp_sorted["ratio"].iloc[-1]),
            "limiting_constraint": str(top["constraint"]),
            "n_excluded": int(counts.get("EXCLUDED", 0)),
            "n_survives": int(counts.get("SURVIVES", 0)),
            "n_unconstrained": int(counts.get("UNCONSTRAINED", 0)),
            "channel_verdicts": json.dumps(verdict_map),
        })
        records.append(row)

    return (
        pd.DataFrame(records)
        .sort_values(group_cols[0])
        .reset_index(drop=True)
    )


def collapse_numeric_metric(
    df: pd.DataFrame,
    group_cols: list[str],
    metric_col: str,
) -> pd.DataFrame:
    """Aggregate a numeric column across grouped rows into summary statistics."""
    if df.empty or not group_cols or metric_col not in df.columns:
        return pd.DataFrame()

    grouped = (
        df.groupby(group_cols, dropna=False)[metric_col]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )
    return grouped.rename(columns={
        "count": f"{metric_col}_count",
        "mean": f"{metric_col}_mean",
        "median": f"{metric_col}_median",
        "std": f"{metric_col}_std",
        "min": f"{metric_col}_min",
        "max": f"{metric_col}_max",
    })


def export_study_bundle(
    raw_rows: list[dict[str, Any]],
    out_prefix: str = "qgp_study_bundle",
    collapse_group_cols: list[str] | None = None,
    numeric_metrics: list[str] | None = None,
) -> dict[str, Any]:
    """Write raw and collapsed tables to both CSV and JSON as a complete study bundle."""
    raw_df = pd.DataFrame(raw_rows)
    raw_json_path = Path(f"{out_prefix}_raw.json")
    raw_csv_path = Path(f"{out_prefix}_raw.csv")
    collapsed_json_path = Path(f"{out_prefix}_collapsed.json")
    collapsed_csv_path = Path(f"{out_prefix}_collapsed.csv")

    raw_df.to_json(raw_json_path, orient="records", indent=2)
    raw_df.to_csv(raw_csv_path, index=False)

    group_cols = collapse_group_cols or [
        c for c in ["scale_over_epl", "eff_scale_gev"] if c in raw_df.columns
    ]
    collapsed_df = collapse_channel_verdicts(raw_df, group_cols=group_cols)

    if not collapsed_df.empty and numeric_metrics:
        for metric in numeric_metrics:
            m_df = collapse_numeric_metric(raw_df, group_cols, metric)
            if not m_df.empty:
                collapsed_df = collapsed_df.merge(m_df, on=group_cols, how="left")

    collapsed_df.to_json(collapsed_json_path, orient="records", indent=2)
    collapsed_df.to_csv(collapsed_csv_path, index=False)

    return {
        "raw_json": str(raw_json_path),
        "raw_csv": str(raw_csv_path),
        "collapsed_json": str(collapsed_json_path),
        "collapsed_csv": str(collapsed_csv_path),
        "raw_rows": int(len(raw_df)),
        "collapsed_rows": int(len(collapsed_df)),
    }


def extract_verdict_transitions(
    collapsed_df: pd.DataFrame,
    raw_df: pd.DataFrame | None = None,
    scale_col: str = "scale_over_epl",
) -> dict[str, list[dict[str, Any]]]:
    """
    Extract exact grid intervals where strictest_verdict (in collapsed_df)
    or individual constraint verdicts (in raw_df) change state.
    """
    strictest_transitions: list[dict[str, Any]] = []
    if not collapsed_df.empty and {scale_col, "strictest_verdict"}.issubset(collapsed_df.columns):
        c_sorted = collapsed_df.sort_values(scale_col).reset_index(drop=True)
        for i in range(1, len(c_sorted)):
            prev_row = c_sorted.iloc[i - 1]
            curr_row = c_sorted.iloc[i]
            if prev_row["strictest_verdict"] != curr_row["strictest_verdict"]:
                strictest_transitions.append({
                    "from_scale": float(prev_row[scale_col]),
                    "to_scale": float(curr_row[scale_col]),
                    "from_verdict": str(prev_row["strictest_verdict"]),
                    "to_verdict": str(curr_row["strictest_verdict"]),
                    "from_max_ratio": float(prev_row["max_ratio"]) if "max_ratio" in curr_row else None,
                    "to_max_ratio": float(curr_row["max_ratio"]) if "max_ratio" in curr_row else None,
                    "limiting_constraint": (
                        str(curr_row["limiting_constraint"])
                        if "limiting_constraint" in curr_row
                        else None
                    ),
                })

    channel_transitions: list[dict[str, Any]] = []
    if (
        raw_df is not None
        and not raw_df.empty
        and {"constraint", scale_col, "verdict"}.issubset(raw_df.columns)
    ):
        r_sorted = raw_df.sort_values(["constraint", scale_col]).reset_index(drop=True)
        for cname, grp in r_sorted.groupby("constraint", dropna=False):
            grp_reset = grp.reset_index(drop=True)
            for i in range(1, len(grp_reset)):
                p = grp_reset.iloc[i - 1]
                q = grp_reset.iloc[i]
                if p["verdict"] != q["verdict"]:
                    channel_transitions.append({
                        "constraint": str(cname),
                        "from_scale": float(p[scale_col]),
                        "to_scale": float(q[scale_col]),
                        "from_verdict": str(p["verdict"]),
                        "to_verdict": str(q["verdict"]),
                        "from_ratio": float(p["ratio"]) if "ratio" in p else None,
                        "to_ratio": float(q["ratio"]) if "ratio" in q else None,
                    })

    return {
        "strictest_transitions": strictest_transitions,
        "channel_transitions": channel_transitions,
    }
