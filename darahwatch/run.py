"""The daily run: forecast the next 14 days for every state and blood group, save the forecast
(never overwriting an earlier one), score every past forecast, and write the bulletin.

    python -m darahwatch.run --data live|snapshot|PATH [--root .]

Files written (all under --root):
  forecasts/<data end>.csv         every daily forecast, all three methods, with 80% ranges
  forecasts/<data end>-weeks.csv   week totals for days 1-7 and 8-14, with ranges and warnings
  latest/forecast.json             what the dashboard reads
  latest/bulletin.md               the bulletin in English and Bahasa Malaysia
  scorecard/scorecard.json         how every saved forecast did, once its days had data
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import bulletin, calendar as cal, data, jsonout, model as M, ranges as R, scorecard

CHOSEN = "Gradient-boosted trees"


def forecast(s: pd.DataFrame, offsets_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    fr = M.Frame.build(s)
    end = fr.dates[-1]
    origins = M.origins_between(fr, "2010-01-01", end - pd.Timedelta(days=M.H), 3)
    model = M.fit(M.rows(fr, origins))
    f = M.rows(fr, np.array([len(fr.dates) - 1]), with_target=False)
    f[CHOSEN] = M.predict(model, f)
    lf = R.long_forecasts(f, M.METHODS)
    daily = R.apply_daily(lf, pd.read_csv(offsets_dir / "daily_offsets.csv"))
    wk = R.apply_weekly(R.weekly(lf), pd.read_csv(offsets_dir / "weekly_offsets.csv"))
    wk = R.warn(wk, R.low_lines(s, (end + pd.Timedelta(days=1)).year))
    return daily, wk


def holidays_ahead(states, start, end) -> list:
    """Public holidays in the forecast window: one row per day and holiday, with its English and
    Malay names (each looked up in the states that have it) and how many states have it."""
    seen = {}
    for d in pd.date_range(start, end):
        for st in states:
            for ms, en in cal.holiday_names(st, d):
                row = seen.setdefault((d, en), {"date": str(d.date()), "name_en": en, "name_ms": ms, "states": 0})
                row["states"] += 1
    return [seen[k] for k in sorted(seen)]


def issue_summary(daily: pd.DataFrame, wk: pd.DataFrame, end: pd.Timestamp) -> dict:
    ch = daily[daily.method == CHOSEN]
    w = wk[wk.method == CHOSEN]
    w1 = ch[ch.h <= 7]
    warnings = [{"state": r.state, "blood_type": r.blood_type, "level": r.warning, "week": int(r.week)}
                for r in w[w.warning != ""].sort_values(["week", "warning", "state", "blood_type"]).itertuples()]
    hol = holidays_ahead(sorted(ch.state.unique()), ch.date.min(), ch.date.max())
    regions = int(ch.state.nunique())
    kl = "W.P. Kuala Lumpur" in set(ch.state)
    where_en = f"{regions - 1} states and Kuala Lumpur" if kl else f"{regions} states"
    where_ms = f"{regions - 1} negeri dan Kuala Lumpur" if kl else f"{regions} negeri"
    return {"data_end": str(end.date()), "regions": regions, "where_en": where_en, "where_ms": where_ms,
            "week1": {"first": str(w1.date.min().date()), "last": str(w1.date.max().date()), "total": float(w1.forecast.sum())},
            "warnings": warnings, "holidays": hol}


def run(source: str, root: Path) -> dict:
    raw = data.load(source)
    s = data.series(raw)
    end = s.date.max()
    daily, wk = forecast(s, root / "evaluation")
    issued = str(end.date())
    fdir = root / "forecasts"
    fdir.mkdir(parents=True, exist_ok=True)
    day_file, week_file = fdir / f"{issued}.csv", fdir / f"{issued}-weeks.csv"
    if not day_file.exists():  # a forecast, once published, is never edited
        cols = ["state", "blood_type", "method", "date", "h", "forecast", "low", "high"]
        out = daily[cols].assign(issued=issued, date=daily.date.dt.date)
        out.round({"forecast": 2, "low": 2, "high": 2}).to_csv(day_file, index=False)
        wcols = ["state", "blood_type", "method", "week", "first", "last", "forecast", "low", "high", "low_line", "warning"]
        wout = wk[wcols].assign(issued=issued, first=wk["first"].dt.date, last=wk["last"].dt.date)
        wout.round({"forecast": 1, "low": 1, "high": 1, "low_line": 1}).to_csv(week_file, index=False)
    summary = issue_summary(daily, wk, end)
    text = bulletin.write({**summary, "holidays": [{"date": h["date"], "name": h["name_en"], "states": h["states"]} for h in summary["holidays"]]})
    text_ms = bulletin.write({**summary, "holidays": [{"date": h["date"], "name": h["name_ms"], "states": h["states"]} for h in summary["holidays"]]})
    card = scorecard.score(fdir, s)
    latest = root / "latest"
    latest.mkdir(parents=True, exist_ok=True)
    ch = daily[daily.method == CHOSEN]
    w = wk[wk.method == CHOSEN]
    payload = {
        "issued": issued, "method": CHOSEN, "range_level": R.LEVEL,
        "calibration_year": int(pd.read_csv(root / "evaluation" / "daily_offsets.csv").calibration_year.iloc[0]),
        "bulletin": {"en": text["en"], "ms": text_ms["ms"]},
        "holidays": summary["holidays"],
        "days": [{"state": r.state, "blood_type": r.blood_type, "date": str(r.date.date()), "h": int(r.h),
                  "forecast": round(float(r.forecast), 1), "low": round(float(r.low), 1), "high": round(float(r.high), 1)}
                 for r in ch.sort_values(["state", "blood_type", "date"]).itertuples()],
        "weeks": [{"state": r.state, "blood_type": r.blood_type, "week": int(r.week), "first": str(r.first.date()),
                   "last": str(r.last.date()), "forecast": round(float(r.forecast), 1), "low": round(float(r.low), 1),
                   "high": round(float(r.high), 1), "low_line": round(float(r.low_line), 1), "warning": r.warning}
                  for r in w.sort_values(["state", "blood_type", "week"]).itertuples()],
        "scorecard": card,
    }
    (latest / "forecast.json").write_text(jsonout.dumps(payload))
    (latest / "bulletin.md").write_text(
        f"# DarahWatch bulletin, {issued}\n\n## English\n\n{text['en']}\n\n## Bahasa Malaysia\n\n{text_ms['ms']}\n")
    (root / "scorecard").mkdir(exist_ok=True)
    (root / "scorecard" / "scorecard.json").write_text(jsonout.dumps(card))
    return payload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="live")
    p.add_argument("--root", default=".")
    a = p.parse_args()
    r = run(a.data, Path(a.root))
    print(r["bulletin"]["en"])
    print(r["bulletin"]["ms"])


if __name__ == "__main__":
    main()
