"""The honest check: rolling backtests, ranges built on one year and scored on the next, and the
low-week rule scored on years it never saw. Writes evaluation/backtest.json and the ranges the
daily forecast uses (evaluation/daily_offsets.csv, evaluation/weekly_offsets.csv).

    python -m darahwatch.evaluate --data snapshot|live|PATH --out evaluation
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, jsonout, model as M, ranges as R


def day_kind(bt: pd.DataFrame) -> np.ndarray:
    near = np.minimum(bt.t_days_to_holiday, bt.t_days_since_holiday)
    return np.select([bt.t_holiday == 1, bt.t_ramadan == 1, near <= 3], ["Public holiday", "Ramadan", "Within 3 days of a holiday"],
                     "Other days")


def mae_table(bt: pd.DataFrame) -> list:
    out = []
    kind = day_kind(bt)
    for m in M.METHODS:
        e = (bt.actual - bt[m]).abs()
        row = {"method": m, "MAE": float(e.mean()), "days_1_7": float(e[bt.h <= 7].mean()), "days_8_14": float(e[bt.h > 7].mean())}
        row["by_day"] = {k: float(e[kind == k].mean()) for k in ["Public holiday", "Ramadan", "Within 3 days of a holiday", "Other days"]}
        out.append(row)
    return out


def run(source: str, out_dir: Path, years=None) -> dict:
    raw = data.load(source)
    s = data.series(raw)
    fr = M.Frame.build(s)
    end = fr.dates[-1]
    last_complete = end.year if end >= pd.Timestamp(f"{end.year}-12-31") else end.year - 1
    years = years or list(range(last_complete - 2, end.year + 1))  # three complete years and this year so far
    bts = {y: M.backtest_year(fr, y) for y in years}
    report = {"data": {**data.audit(raw), "analysed_to": str(end.date()), "missing_state_group_days": data.missing_days(s)},
              "methods": M.METHODS, "range_level": R.LEVEL, "low_quantile": R.LOW_QUANTILE, "years": []}
    for y in years:
        bt = bts[y]
        row = {"year": int(y), "partial": bool(y > last_complete), "forecasts": int(len(bt)), "mae": mae_table(bt)}
        if y - 1 in bts:
            cal = R.long_forecasts(bts[y - 1], M.METHODS)
            lf = R.long_forecasts(bt, M.METHODS)
            d = R.apply_daily(lf, R.daily_offsets(cal))
            d["inside"] = (d.actual >= d.low) & (d.actual <= d.high)
            row["daily_range"] = [{"method": m, "coverage": float(g.inside.mean()), "mean_width": float((g.high - g.low).mean())}
                                  for m, g in d.groupby("method", sort=False)]
            wk = R.apply_weekly(R.weekly(lf), R.weekly_offsets(R.weekly(cal)))
            wk = wk[wk.actual.notna()]
            wk["inside"] = (wk.actual >= wk.low) & (wk.actual <= wk.high)
            row["weekly_range"] = [{"method": m, "coverage": float(g.inside.mean()), "mean_width": float((g.high - g.low).mean())}
                                   for m, g in wk.groupby("method", sort=False)]
            w = R.warn(wk, R.low_lines(s, y))
            row["low_week_rule"] = {m: R.warning_scores(g) for m, g in w.groupby("method", sort=False)}
            best = w[w.method == M.METHODS[-1]]
            row["by_state"] = []
            for st, g in d[d.method == M.METHODS[-1]].groupby("state"):
                ws = R.warning_scores(best[best.state == st])
                simple = d[(d.method == M.BEST_SIMPLE) & (d.state == st)]
                row["by_state"].append({"state": st, "coverage": float(g.inside.mean()),
                                        "MAE": float((g.actual - g.forecast).abs().mean()),
                                        "MAE_simple": float((simple.actual - simple.forecast).abs().mean()),
                                        "weeks": ws["weeks"], "really_low_weeks": ws["really_low_weeks"],
                                        "warnings": ws["any_warning"]["flagged"], "warning_hit_rate": ws["any_warning"]["hit_rate"],
                                        "warning_catch_rate": ws["any_warning"]["catch_rate"]})
        report["years"].append(row)
    # The ranges the daily forecast uses: built on the latest complete year.
    out_dir.mkdir(parents=True, exist_ok=True)
    cal = R.long_forecasts(bts[last_complete], M.METHODS)
    R.daily_offsets(cal).assign(calibration_year=last_complete).to_csv(out_dir / "daily_offsets.csv", index=False)
    R.weekly_offsets(R.weekly(cal)).assign(calibration_year=last_complete).to_csv(out_dir / "weekly_offsets.csv", index=False)
    report["calibration_year"] = int(last_complete)
    (out_dir / "backtest.json").write_text(jsonout.dumps(report))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="live")
    p.add_argument("--out", default="evaluation")
    a = p.parse_args()
    r = run(a.data, Path(a.out))
    for y in r["years"]:
        print(y["year"], {m["method"]: round(m["MAE"], 3) for m in y["mae"]},
              {c["method"]: round(c["coverage"], 3) for c in y.get("daily_range", [])})


if __name__ == "__main__":
    main()
