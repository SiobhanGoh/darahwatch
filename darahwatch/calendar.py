"""Calendar signals for each state and day: public holidays, festive seasons and Ramadan.

Public holidays come from the `holidays` package (state by state, including state-only days such
as Hari Hol in Pahang or the Sultan's birthday). Ramadan starts on 1 Ramadan of the Umm al-Qura
calendar (`hijridate`) and ends the day before Hari Raya Puasa as Malaysia observed it, so the
festive day itself is always Malaysia's own date.
"""
from __future__ import annotations

from functools import lru_cache

import holidays
import numpy as np
import pandas as pd
from hijridate import Gregorian, Hijri

# data.gov.my state names -> ISO 3166-2:MY codes used by `holidays`.
STATE_CODES = {
    "Johor": "01", "Kedah": "02", "Kelantan": "03", "Melaka": "04", "Negeri Sembilan": "05",
    "Pahang": "06", "Pulau Pinang": "07", "Perak": "08", "Perlis": "09", "Selangor": "10",
    "Terengganu": "11", "Sabah": "12", "Sarawak": "13", "W.P. Kuala Lumpur": "14", "W.P. Labuan": "15",
    "W.P. Putrajaya": "16",
}
# Which festive season a holiday belongs to, by the words in its Malay name.
KINDS = [("raya_puasa", "Hari Raya Puasa"), ("raya_haji", "Hari Raya Haji"), ("cny", "Tahun Baharu Cina"),
         ("deepavali", "Deepavali"), ("christmas", "Krismas"), ("gawai", "Gawai"), ("kaamatan", "Kaamatan")]
KIND_CODES = {k: i + 1 for i, (k, _) in enumerate(KINDS)}  # 0 = no holiday, len+1 = other holiday
WINDOW = 14  # days counted before or after a holiday; further away counts as WINDOW + 1


def kind_of(name: str) -> int:
    """The festive season of a holiday name; several holidays on one day are joined by '; '."""
    for k, words in KINDS:  # in order, so Raya wins over a national day falling on it
        if words.lower() in name.lower():
            return KIND_CODES[k]
    return len(KINDS) + 1


@lru_cache(maxsize=None)
def _holidays(state: str, first: int, last: int) -> dict:
    code = STATE_CODES.get(state)
    h = holidays.Malaysia(subdiv=code, years=range(first, last + 1)) if code else holidays.Malaysia(years=range(first, last + 1))
    return {pd.Timestamp(d): name for d, name in h.items()}


@lru_cache(maxsize=None)
def ramadan_days(first: int, last: int) -> frozenset:
    raya = sorted(pd.Timestamp(d) for d, n in holidays.Malaysia(years=range(first, last + 2)).items()
                  if "Hari Raya Puasa" in n.split("; "))
    out = set()
    for r in raya:
        h = Gregorian(r.year, r.month, r.day).to_hijri()
        start = Hijri(h.year if h.month >= 10 else h.year - 1, 9, 1).to_gregorian()
        # Never longer than 30 days: where the Umm al-Qura start is a day early for Malaysia.
        begin = max(pd.Timestamp(start.year, start.month, start.day), r - pd.Timedelta(days=30))
        out.update(pd.date_range(begin, r - pd.Timedelta(days=1)))
    return frozenset(out)


def _distance(days: pd.DatetimeIndex, marks: list, forward: bool) -> np.ndarray:
    """Days until the next (forward) or since the last mark, capped at WINDOW + 1."""
    out = np.full(len(days), WINDOW + 1, dtype=float)
    if not marks:
        return out
    m = np.array(sorted(marks), dtype="datetime64[D]")
    t = days.values.astype("datetime64[D]")
    if forward:
        i = np.searchsorted(m, t, side="left")
        ok = i < len(m)
        out[ok] = (m[i[ok]] - t[ok]).astype(int)
    else:
        i = np.searchsorted(m, t, side="right") - 1
        ok = i >= 0
        out[ok] = (t[ok] - m[i[ok]]).astype(int)
    return np.minimum(out, WINDOW + 1)


def features(state: str, days: pd.DatetimeIndex) -> pd.DataFrame:
    """One row per day: weekday, month, holiday flags and distances, Ramadan."""
    days = pd.DatetimeIndex(days)
    first, last = days.min().year - 1, days.max().year + 1
    hol = _holidays(state, first, last)
    names = [hol.get(d, "") for d in days]
    raya = [d for d, n in hol.items() if "Hari Raya Puasa" in n.split("; ")]
    ram = ramadan_days(first, last)
    in_ram = np.array([d in ram for d in days])
    # Day of Ramadan (1-30), counted from the first Ramadan day in a run of consecutive days.
    ram_day = np.zeros(len(days))
    for i in range(len(days)):
        if in_ram[i]:
            follows = i > 0 and in_ram[i - 1] and (days[i] - days[i - 1]).days == 1
            ram_day[i] = ram_day[i - 1] + 1 if follows else 1
    to_raya = _distance(days, raya, forward=True)
    since_raya = _distance(days, raya, forward=False)
    return pd.DataFrame({
        "date": days,
        "weekday": days.dayofweek,
        "month": days.month,
        "holiday": np.array([n != "" for n in names], dtype=int),
        "holiday_kind": [kind_of(n) if n else 0 for n in names],
        "days_to_holiday": _distance(days, list(hol), forward=True),
        "days_since_holiday": _distance(days, list(hol), forward=False),
        "ramadan": in_ram.astype(int),
        "ramadan_day": ram_day,
        "days_to_raya": to_raya,
        "days_since_raya": since_raya,
    })


def holiday_name(state: str, day: pd.Timestamp) -> str:
    day = pd.Timestamp(day)
    return _holidays(state, day.year - 1, day.year + 1).get(day, "")
