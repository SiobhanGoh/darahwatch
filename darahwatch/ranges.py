"""80% ranges from past mistakes (split conformal), and the low-week warning rule.

Formula. For one state, blood group, forecaster and look-ahead, collect the errors e = actual -
forecast from one complete backtest year (n days), sorted from smallest to largest. With
alpha = 0.2 (an 80% range):

    low  = forecast + e_(lo),   lo = floor((n + 1) * alpha / 2)
    high = forecast + e_(hi),   hi = ceil((n + 1) * (1 - alpha / 2))

where e_(k) is the k-th smallest error. Counts cannot be negative, so low is cut at 0. The ranges
are built on year Y - 1 and checked on year Y, which they never saw.

THE LOW-WEEK RULE (fixed; written down before any test year was scored, and never changed):
for each state and blood group, a "normal week" is any run of 7 days in the three calendar years
before the forecast's year. The reference line is the 30th percentile of those 7-day totals.
For the next 7 days, and for the 7 days after:
  * "Likely low"  when the HIGH end of the 80% range for the week's total is below the line;
  * "Watch"       when the forecast itself is below the line (but the high end is not);
  * otherwise no warning.
It is about donations collected, not blood stock or hospital demand.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

LEVEL = 0.8
LOW_QUANTILE = 0.30
KEYS = ["state", "blood_type", "method"]


def conformal(errors, level: float = LEVEL) -> tuple[float, float]:
    e = np.sort(np.asarray(errors, dtype=float)[np.isfinite(errors)])
    n = len(e)
    if n < 10:
        return (np.nan, np.nan)
    a = (1 - level) / 2
    lo = max(1, math.floor(round((n + 1) * a, 8)))
    hi = min(n, math.ceil(round((n + 1) * (1 - a), 8)))
    return float(e[lo - 1]), float(e[hi - 1])


def bucket(h) -> np.ndarray:
    return np.where(np.asarray(h) <= 7, "days 1-7", "days 8-14")


def long_forecasts(bt: pd.DataFrame, methods) -> pd.DataFrame:
    """One row per forecast and method: state, blood_type, method, origin, date, h, forecast, actual."""
    keep = ["state", "blood_type", "origin", "date", "h", "actual"]
    out = bt[keep + list(methods)].melt(id_vars=keep, var_name="method", value_name="forecast")
    return out.dropna(subset=["forecast"])


def daily_offsets(lf: pd.DataFrame) -> pd.DataFrame:
    lf = lf.assign(bucket=bucket(lf.h), error=lf.actual - lf.forecast)
    rows = [dict(zip(KEYS + ["bucket"], k), lo=o[0], hi=o[1], n=len(g))
            for k, g in lf.groupby(KEYS + ["bucket"]) for o in [conformal(g.error)]]
    return pd.DataFrame(rows)


def apply_daily(lf: pd.DataFrame, off: pd.DataFrame) -> pd.DataFrame:
    x = lf.assign(bucket=bucket(lf.h)).merge(off[KEYS + ["bucket", "lo", "hi"]], on=KEYS + ["bucket"], how="left")
    x["low"] = np.maximum(0, x.forecast + x.lo)
    x["high"] = np.maximum(x.low, x.forecast + x.hi)
    return x.drop(columns=["lo", "hi", "bucket"])


def weekly(lf: pd.DataFrame) -> pd.DataFrame:
    """Totals for days 1-7 ("week 1") and 8-14 ("week 2") of each forecast run. Only complete weeks."""
    x = lf.assign(week=np.where(lf.h <= 7, 1, 2))
    g = x.groupby(KEYS + ["origin", "week"])
    out = g.agg(forecast=("forecast", "sum"), actual=("actual", "sum"), known=("actual", "count"),
                days=("h", "size"), first=("date", "min"), last=("date", "max")).reset_index()
    out.loc[out.known < 7, "actual"] = np.nan  # a week with any unknown day has no real total yet
    return out[out.days == 7].drop(columns=["days", "known"])


def weekly_offsets(wk: pd.DataFrame) -> pd.DataFrame:
    wk = wk.assign(error=wk.actual - wk.forecast)
    rows = [dict(zip(KEYS + ["week"], k), lo=o[0], hi=o[1], n=len(g))
            for k, g in wk.groupby(KEYS + ["week"]) for o in [conformal(g.error)]]
    return pd.DataFrame(rows)


def apply_weekly(wk: pd.DataFrame, off: pd.DataFrame) -> pd.DataFrame:
    x = wk.merge(off[KEYS + ["week", "lo", "hi"]], on=KEYS + ["week"], how="left")
    x["low"] = np.maximum(0, x.forecast + x.lo)
    x["high"] = np.maximum(x.low, x.forecast + x.hi)
    return x.drop(columns=["lo", "hi"])


def low_lines(s: pd.DataFrame, year: int) -> pd.DataFrame:
    """The 30th percentile of every 7-day total in the three calendar years before `year`."""
    x = s[(s.date.dt.year >= year - 3) & (s.date.dt.year < year)].sort_values("date")
    tot = x.groupby(["state", "blood_type"]).donations.transform(lambda v: v.rolling(7, min_periods=7).sum())
    x = x.assign(week_total=tot).dropna(subset=["week_total"])
    return (x.groupby(["state", "blood_type"]).week_total.quantile(LOW_QUANTILE)
            .rename("low_line").reset_index().assign(line_year=year))


def warn(wk: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    x = wk.merge(lines[["state", "blood_type", "low_line"]], on=["state", "blood_type"], how="left")
    x["warning"] = np.where(x.high < x.low_line, "Likely low", np.where(x.forecast < x.low_line, "Watch", ""))
    return x


def warning_scores(x: pd.DataFrame) -> dict:
    """How the rule did where the real week total is known: hit rate and catch rate per level."""
    x = x[x.actual.notna()]
    low = x.actual < x.low_line
    out = {"weeks": int(len(x)), "really_low_weeks": int(low.sum()), "base_rate": float(low.mean())}
    for name, flag in [("likely_low", x.warning == "Likely low"), ("any_warning", x.warning != "")]:
        out[name] = {"flagged": int(flag.sum()),
                     "hit_rate": float(low[flag].mean()) if flag.any() else None,
                     "catch_rate": float(flag[low].mean()) if low.any() else None}
    return out
