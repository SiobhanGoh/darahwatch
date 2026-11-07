"""Forecasts of daily donations 1 to 14 days ahead, for every state and blood group.

Three forecasters, compared on the same backtests:

* Same weekday last week ("seasonal naive"): the most recent count for the target's weekday that
  was already known on the forecast day. The bar every other method has to clear.
* Recent weekday average: the mean of the target's weekday over the last 156 weeks (three years)
  known on the forecast day. This is the Klang Valley dashboard's best method.
* Gradient-boosted trees (LightGBM, Poisson loss), one model for all states and groups. It sees
  the same weekday averages plus the calendar: public holidays (state by state), festive seasons,
  Ramadan, and how far each day is from them. That is what the two simple methods cannot know.

Every feature for a forecast made on day `o` uses counts up to and including `o` only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import calendar as cal
from .data import GROUPS, wide

H = 14  # days ahead
METHODS = ["Same weekday last week", "Recent weekday average", "Gradient-boosted trees"]
BEST_SIMPLE = "Recent weekday average"
CAL_COLS = ["weekday", "month", "holiday", "holiday_kind", "days_to_holiday", "days_since_holiday",
            "ramadan", "ramadan_day", "days_to_raya", "days_since_raya"]
FEATURES = ["state_id", "group_id", "h", "lvl7", "lvl28", "lvl91", "wd1", "wd4", "wd13", "wd52", "wd156",
            "ly7", "hol_last7", "ram_last7"] + [f"t_{c}" for c in CAL_COLS]
CATEGORICAL = ["state_id", "group_id", "t_holiday_kind"]
PARAMS = dict(objective="poisson", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, seed=7, num_threads=4, deterministic=True, force_row_wise=True)
ROUNDS = 500


def _nanmean_windows(x: np.ndarray, width: int, step: int) -> np.ndarray:
    """out[i] = mean of x[i], x[i-step], ..., x[i-step*(width-1)], ignoring NaN; NaN before enough history."""
    n = len(x)
    out = np.full(n, np.nan)
    v = np.nan_to_num(x)
    ok = (~np.isnan(x)).astype(float)
    for r in range(step):
        idx = np.arange(r, n, step)
        cs = np.concatenate([[0.0], np.cumsum(v[idx])])
        cn = np.concatenate([[0.0], np.cumsum(ok[idx])])
        k = np.arange(1, len(idx) + 1)
        lo = np.maximum(k - width, 0)
        s, c = cs[k] - cs[lo], cn[k] - cn[lo]
        full = k >= width
        res = np.where(full & (c > 0), s / np.maximum(c, 1), np.nan)
        out[idx] = res
    return out


@dataclass
class Frame:
    """Everything the features are cut from: the counts as a dates x series matrix, plus calendars."""
    dates: pd.DatetimeIndex
    keys: list            # (state, blood_type) per column
    y: np.ndarray         # dates x series
    calendar: dict        # state -> DataFrame of calendar features over `dates` + H days
    windows: dict         # name -> dates x series matrix of window means

    @classmethod
    def build(cls, s: pd.DataFrame) -> "Frame":
        w = wide(s)
        dates = pd.DatetimeIndex(w.index)
        y = w.to_numpy(dtype=float)
        ext = pd.date_range(dates[0], dates[-1] + pd.Timedelta(days=H), freq="D")
        states = sorted({k[0] for k in w.columns})
        calendar = {st: cal.features(st, ext).set_index("date") for st in states}
        win = {}
        for name, width, step in [("lvl7", 7, 1), ("lvl28", 28, 1), ("lvl91", 91, 1), ("wd1", 1, 7), ("wd4", 4, 7),
                                  ("wd13", 13, 7), ("wd52", 52, 7), ("wd156", 156, 7)]:
            win[name] = np.column_stack([_nanmean_windows(y[:, j], width, step) for j in range(y.shape[1])])
        return cls(dates, list(w.columns), y, calendar, win)

    def index_of(self, day) -> int:
        return int(self.dates.get_loc(pd.Timestamp(day)))


def rows(fr: Frame, origins: np.ndarray, horizons=range(1, H + 1), with_target: bool = True) -> pd.DataFrame:
    """Feature rows for every series, origin index and horizon. Targets past the data are NaN."""
    states = sorted({k[0] for k in fr.keys})
    sid = {s: i for i, s in enumerate(states)}
    out = []
    n = len(fr.dates)
    for j, (state, group) in enumerate(fr.keys):
        c = fr.calendar[state]
        hol = c["holiday"].to_numpy()
        ram = c["ramadan"].to_numpy()
        hol7 = pd.Series(hol).rolling(7, min_periods=1).sum().to_numpy()
        ram7 = pd.Series(ram).rolling(7, min_periods=1).sum().to_numpy()
        cal_arr = {k: c[k].to_numpy() for k in CAL_COLS}
        for h in horizons:
            o = origins
            t = o + h
            last_same = t - 7 * int(np.ceil(h / 7))  # newest day with the target's weekday known at o
            d = {
                "state": state, "blood_type": group, "state_id": sid[state], "group_id": GROUPS.index(group),
                "origin": fr.dates[o], "date": fr.dates[0] + pd.to_timedelta(t, unit="D"), "h": h,
                "lvl7": fr.windows["lvl7"][o, j], "lvl28": fr.windows["lvl28"][o, j], "lvl91": fr.windows["lvl91"][o, j],
                "hol_last7": hol7[o], "ram_last7": ram7[o],
            }
            for name in ["wd1", "wd4", "wd13", "wd52", "wd156"]:
                d[name] = fr.windows[name][last_same, j]
            ly = t - 364  # same weekday a year earlier: always known for h <= 14
            d["ly7"] = np.where(ly - 3 >= 0, fr.windows["lvl7"][np.clip(ly + 3, 0, n - 1), j], np.nan)
            for k in CAL_COLS:
                d[f"t_{k}"] = cal_arr[k][t]
            d["actual"] = np.where(t < n, fr.y[np.clip(t, 0, n - 1), j], np.nan) if with_target else np.nan
            out.append(pd.DataFrame(d))
    df = pd.concat(out, ignore_index=True)
    df["Same weekday last week"] = df["wd1"]
    df["Recent weekday average"] = df["wd156"]
    return df


def fit(train: pd.DataFrame):
    import lightgbm as lgb
    tr = train.dropna(subset=["actual", "wd156", "ly7"])
    ds = lgb.Dataset(tr[FEATURES], tr["actual"], categorical_feature=CATEGORICAL, free_raw_data=False)
    return lgb.train(PARAMS, ds, num_boost_round=ROUNDS)


def predict(model, df: pd.DataFrame) -> np.ndarray:
    return np.maximum(model.predict(df[FEATURES]), 0.0)


def origins_between(fr: Frame, first, last, step: int = 1) -> np.ndarray:
    a, b = fr.index_of(first), fr.index_of(last)
    return np.arange(a, b + 1, step)


def train_origins(fr: Frame, before_year: int, step: int = 3) -> np.ndarray:
    """Origins from 2010 (three years of history for the weekday averages) whose targets all fall
    before `before_year`, every `step` days so the origin's weekday varies."""
    last = pd.Timestamp(f"{before_year - 1}-12-31") - pd.Timedelta(days=H)
    return origins_between(fr, "2010-01-01", min(last, fr.dates[-1] - pd.Timedelta(days=H)), step)


def backtest_year(fr: Frame, year: int, step_test: int = 1) -> pd.DataFrame:
    """Pretend it is each day of `year` in turn: forecast the next 14 days with a model trained only
    on targets before the year began, and record what really happened."""
    model = fit(rows(fr, train_origins(fr, year)))
    first = pd.Timestamp(f"{year}-01-01") - pd.Timedelta(days=1)  # forecasts made on 31 Dec cover 1-14 Jan
    last = min(pd.Timestamp(f"{year}-12-31"), fr.dates[-1]) - pd.Timedelta(days=1)
    test = rows(fr, origins_between(fr, first, last, step_test))
    test = test[(test.date.dt.year == year) & test.actual.notna()].copy()
    test["Gradient-boosted trees"] = predict(model, test)
    test["year"] = year
    return test
