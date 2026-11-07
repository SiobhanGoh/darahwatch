import json
import math

import numpy as np
import pandas as pd
import pytest

from darahwatch import bulletin, calendar as cal, data, model as M, ranges as R, scorecard


@pytest.fixture(scope="module")
def snapshot():
    return data.load_snapshot()


@pytest.fixture(scope="module")
def small(snapshot):
    """Two states, 2015 to mid 2026: enough history for every feature, quick to model."""
    d = snapshot[snapshot.state.isin(["Kelantan", "W.P. Kuala Lumpur"])]
    return data.series(d)


def test_snapshot_checks(snapshot):
    a = data.audit(snapshot)
    assert a["duplicate_keys"] == 0 and a["negative_counts"] == 0 and a["missing_values"] == 0
    assert a["states"] == 13 and a["last_date"] == "2026-08-28"
    assert a["all_not_equal_to_sum_of_groups"] == 280  # see DATA_NOTES.md


def test_series_is_a_full_calendar(small):
    n_days = (small.date.max() - small.date.min()).days + 1
    assert len(small) == 2 * 4 * n_days
    assert data.missing_days(small) == 0


def test_conformal_ranks_follow_the_formula():
    e = np.arange(1, 20)  # n = 19: lo rank floor(20 * 0.1) = 2, hi rank ceil(20 * 0.9) = 18
    assert R.conformal(e) == (2.0, 18.0)
    assert all(math.isnan(v) for v in R.conformal(np.arange(5)))


def test_ramadan_and_raya():
    f = cal.features("Selangor", pd.date_range("2025-02-25", "2025-04-02"))
    ram = f[f.ramadan == 1]
    assert ram.date.min() == pd.Timestamp("2025-03-01") and ram.date.max() == pd.Timestamp("2025-03-30")
    assert ram.ramadan_day.max() == 30
    raya = f[f.date == pd.Timestamp("2025-03-31")].iloc[0]
    assert raya.holiday == 1 and raya.holiday_kind == cal.KIND_CODES["raya_puasa"] and raya.days_to_raya == 0


def test_state_holidays_differ():
    # Hari Hol Pahang is a Pahang holiday only.
    assert cal.holiday_name("Pahang", "2025-05-22") != ""
    assert cal.holiday_name("Selangor", "2025-05-22") == ""


def test_features_use_only_what_was_known_on_the_forecast_day(small):
    fr = M.Frame.build(small)
    o = fr.index_of("2025-06-01")
    before = M.rows(fr, np.array([o]))
    # Scramble every count after the forecast day: no feature may change.
    changed = small.copy()
    after = changed.date > pd.Timestamp("2025-06-01")
    changed.loc[after, "donations"] = np.random.default_rng(0).integers(0, 999, after.sum())
    fr2 = M.Frame.build(changed)
    again = M.rows(fr2, np.array([o]))
    pd.testing.assert_frame_equal(before[M.FEATURES], again[M.FEATURES])
    assert not before.actual.equals(again.actual)  # the targets did change


def test_simple_methods_match_their_definitions(small):
    fr = M.Frame.build(small)
    o = fr.index_of("2025-06-01")
    r = M.rows(fr, np.array([o]), horizons=[3, 10])
    w = data.wide(small)
    for row in r.itertuples():
        col = w[(row.state, row.blood_type)]
        same = [d for d in col.index if d <= pd.Timestamp("2025-06-01") and d.dayofweek == row.date.dayofweek]
        assert row._asdict()["wd1"] == col[same[-1]]
        assert abs(row.wd156 - col[same[-156:]].mean()) < 1e-9


def test_weekly_totals_and_warning_rule():
    lf = pd.DataFrame({"state": "X", "blood_type": "O", "method": "m", "origin": pd.Timestamp("2025-01-01"),
                       "date": pd.date_range("2025-01-02", periods=14), "h": range(1, 15),
                       "forecast": [10.0] * 14, "actual": [9.0] * 7 + [np.nan] + [20.0] * 6})
    wk = R.weekly(lf)
    assert list(wk.forecast) == [70.0, 70.0] and wk.actual.iloc[0] == 63.0 and np.isnan(wk.actual.iloc[1])
    off = pd.DataFrame({"state": "X", "blood_type": "O", "method": "m", "week": [1, 2], "lo": -5.0, "hi": 5.0})
    x = R.warn(R.apply_weekly(wk, off), pd.DataFrame({"state": ["X"], "blood_type": ["O"], "low_line": [76.0]}))
    assert list(x.warning) == ["Likely low", "Likely low"]  # high end 75 < 76
    x = R.warn(R.apply_weekly(wk, off), pd.DataFrame({"state": ["X"], "blood_type": ["O"], "low_line": [72.0]}))
    assert list(x.warning) == ["Watch", "Watch"]


def test_bulletin_only_uses_the_numbers_given():
    issue = {"data_end": "2026-08-28", "week1": {"first": "2026-08-29", "last": "2026-09-04", "total": 11293.4},
             "warnings": [{"state": "Kelantan", "blood_type": "AB", "level": "Likely low", "week": 1}],
             "holidays": [{"date": "2026-08-31", "name": "Hari Kebangsaan", "states": 13}]}
    b = bulletin.write(issue)
    assert "11,293" in b["en"] and "11,293" in b["ms"]
    assert "Kelantan AB" in b["en"] and "Kelantan AB" in b["ms"]
    assert "Isnin 31 Ogos 2026" in b["ms"] and "semua negeri" in b["ms"]


def test_scorecard_scores_saved_forecasts(tmp_path, small):
    folder = tmp_path / "forecasts"
    folder.mkdir()
    pd.DataFrame({"state": "Kelantan", "blood_type": "O", "method": "m", "date": ["2026-08-01", "2026-08-02"],
                  "h": [1, 2], "forecast": [50.0, 50.0], "low": [40.0, 40.0], "high": [60.0, 60.0],
                  "issued": "2026-07-31"}).to_csv(folder / "2026-07-31.csv", index=False)
    real = small[(small.state == "Kelantan") & (small.blood_type == "O") & small.date.isin(pd.to_datetime(["2026-08-01", "2026-08-02"]))]
    c = scorecard.score(folder, small)
    m = c["methods"][0]
    assert c["scored_days"] == 2 and m["scored"] == 2
    assert abs(m["MAE"] - (real.donations - 50).abs().mean()) < 1e-9
    assert m["coverage"] == ((real.donations >= 40) & (real.donations <= 60)).mean()
