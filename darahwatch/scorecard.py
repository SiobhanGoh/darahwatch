"""The public scorecard: every saved forecast compared with what really happened.

It reads every file in forecasts/ (they are never edited, so the git history of that folder is the
proof of when each forecast was made), joins the real counts once they exist, and reports the
average miss, how often the real count fell inside the 80% range, and how the low-week warnings did.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _read(folder: Path, weeks: bool) -> pd.DataFrame:
    files = sorted(p for p in folder.glob("*.csv") if p.stem.endswith("-weeks") == weeks)
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(p) for p in files], ignore_index=True)


def score(folder: Path, s: pd.DataFrame) -> dict:
    days = _read(folder, weeks=False)
    out = {"forecast_files": len(list(folder.glob("*-weeks.csv"))), "scored_days": 0, "methods": []}
    if days.empty:
        return out
    actual = s[["state", "blood_type", "date", "donations"]].copy()
    actual["date"] = actual.date.dt.strftime("%Y-%m-%d")
    days["date"] = days.date.astype(str)
    j = days.merge(actual, on=["state", "blood_type", "date"], how="inner").dropna(subset=["donations"])
    out["first_issue"], out["last_issue"] = str(days.issued.min()), str(days.issued.max())
    out["scored_days"] = int(j[["issued", "state", "blood_type", "date"]].drop_duplicates().shape[0])
    for m, g in j.groupby("method", sort=False):
        inside = (g.donations >= g.low) & (g.donations <= g.high)
        out["methods"].append({"method": m, "scored": int(len(g)), "MAE": float((g.donations - g.forecast).abs().mean()),
                               "coverage": float(inside.mean()),
                               "from": str(g.date.min()), "to": str(g.date.max())})
    weeks = _read(folder, weeks=True)
    if not weeks.empty:
        # Real week totals: sum the real counts between first and last (inclusive), only when all 7 are known.
        a = s.set_index(["state", "blood_type", "date"]).donations
        def real(r):
            idx = pd.date_range(r["first"], r["last"])
            try:
                v = a.loc[(r["state"], r["blood_type"])].reindex(idx)
            except KeyError:
                return np.nan
            return float(v.sum()) if v.notna().all() and len(v) == 7 else np.nan
        weeks["actual"] = weeks.apply(real, axis=1)
        w = weeks.dropna(subset=["actual"])
        w = w[w.method == "Gradient-boosted trees"]
        if len(w):
            low = w.actual < w.low_line
            flagged = w.warning.fillna("") != ""
            out["weeks"] = {"scored": int(len(w)), "really_low": int(low.sum()), "warnings": int(flagged.sum()),
                            "warning_hit_rate": float(low[flagged].mean()) if flagged.any() else None,
                            "warning_catch_rate": float(flagged[low].mean()) if low.any() else None,
                            "coverage": float(((w.actual >= w.low) & (w.actual <= w.high)).mean())}
    return out
