"""Load, check and reshape Malaysia's daily blood donation counts.

Source: Government of Malaysia, "Daily Blood Donations by Blood Group & State", data.gov.my,
CC BY 4.0. One row per date, state and blood group (A, B, AB, O and "all"). The counts come from
the Blood Bank Information System of the Ministry of Health's main collection sites.

Every check that changes or drops data is described in DATA_NOTES.md.
"""
from __future__ import annotations

import hashlib
import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

LIVE_URL = "https://storage.data.gov.my/healthcare/blood_donations_state.csv"
ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "data" / "blood_donations_state_snapshot.csv.gz"
SNAPSHOT_MD5 = "b5365310d6819b52d39ddeba125093e5"  # of the uncompressed CSV
GROUPS = ["A", "AB", "B", "O"]
START = pd.Timestamp("2006-01-01")


def read_csv_bytes(raw: bytes) -> pd.DataFrame:
    d = pd.read_csv(io.BytesIO(raw), dtype={"state": str, "blood_type": str})
    missing = {"date", "state", "blood_type", "donations"} - set(d.columns)
    if missing:
        raise ValueError(f"The CSV is missing columns: {sorted(missing)}")
    d["date"] = pd.to_datetime(d["date"])
    d["blood_type"] = d["blood_type"].str.upper()
    return d


def load_snapshot(path: Path = SNAPSHOT) -> pd.DataFrame:
    import gzip
    raw = gzip.decompress(Path(path).read_bytes())
    if hashlib.md5(raw).hexdigest() != SNAPSHOT_MD5:
        raise ValueError("The snapshot does not match its recorded checksum.")
    return read_csv_bytes(raw)


def load_live(url: str = LIVE_URL, timeout: int = 120) -> pd.DataFrame:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        raw = r.read()
    if len(raw) < 1_000_000:
        raise ValueError(f"The download from {url} is too small ({len(raw)} bytes) to be the full file.")
    return read_csv_bytes(raw)


def load(source: str) -> pd.DataFrame:
    """'live' downloads the official file; 'snapshot' uses the frozen copy; anything else is a path."""
    if source == "live":
        return load_live()
    if source == "snapshot":
        return load_snapshot()
    return read_csv_bytes(Path(source).read_bytes())


def audit(d: pd.DataFrame) -> dict:
    """The checks DATA_NOTES.md describes. None of them change the data."""
    groups = d[d.blood_type.isin(GROUPS)].groupby(["date", "state"]).donations.sum()
    total = d[d.blood_type == "ALL"].set_index(["date", "state"]).donations
    both = pd.concat([groups.rename("groups"), total.rename("all")], axis=1).dropna()
    return {
        "rows": int(len(d)),
        "first_date": str(d.date.min().date()),
        "last_date": str(d.date.max().date()),
        "states": int(d.state.nunique()),
        "missing_values": int(d.isna().sum().sum()),
        "duplicate_keys": int(d.duplicated(["date", "state", "blood_type"]).sum()),
        "negative_counts": int((d.donations < 0).sum()),
        "all_not_equal_to_sum_of_groups": int((both["all"] != both["groups"]).sum()),
    }


def complete_end(d: pd.DataFrame) -> pd.Timestamp:
    """The newest date on which every state reports all four blood groups."""
    g = d[d.blood_type.isin(GROUPS)]
    need = g.state.nunique() * len(GROUPS)
    n = g.groupby("date").size()
    return n[n == need].index.max()


def series(d: pd.DataFrame, end: pd.Timestamp | None = None) -> pd.DataFrame:
    """Long table: state, blood_type, date, donations, for the four groups, 2006 to `end`.

    Stops on duplicate keys or negative counts rather than guessing. A state-group day that is
    missing inside the calendar is filled with NaN (and reported), never with a made-up count.
    """
    a = audit(d)
    if a["duplicate_keys"] or a["negative_counts"]:
        raise ValueError(f"The data failed its checks: {a}")
    end = complete_end(d) if end is None else pd.Timestamp(end)
    g = d[d.blood_type.isin(GROUPS) & (d.date >= START) & (d.date <= end)]
    days = pd.date_range(START, end, freq="D")
    idx = pd.MultiIndex.from_product([sorted(g.state.unique()), GROUPS, days], names=["state", "blood_type", "date"])
    out = g.set_index(["state", "blood_type", "date"]).donations.reindex(idx).reset_index()
    out["donations"] = out["donations"].astype(float)
    return out


def wide(s: pd.DataFrame) -> pd.DataFrame:
    """Dates as rows, one column per (state, blood_type)."""
    return s.pivot_table(index="date", columns=["state", "blood_type"], values="donations", dropna=False).sort_index(axis=1)


def missing_days(s: pd.DataFrame) -> int:
    return int(np.isnan(s.donations.to_numpy()).sum())
