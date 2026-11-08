# DarahWatch

**Every morning, DarahWatch forecasts blood donations for the next 14 days for each blood group in 12 Malaysian states and Kuala Lumpur (all the places in the official data), puts an honest 80% range on each number, warns when a week is likely to be low, and scores every forecast it has ever published against what really happened.**

It runs for free on GitHub Actions from the Government of Malaysia's open data. Its results appear on the [Klang Valley blood donation dashboard](https://github.com/SiobhanGoh/klang-valley-blood-donation-dashboard) under **Every state**.

> It forecasts **donations collected**. It does not know blood stock, hospital demand or clinical need, so a "low week" is a prompt to look closer, not a shortage alert.

---

## What it found (rolling backtests, 12 states and Kuala Lumpur x 4 blood groups)

"Pretend it is each day of the year in turn, forecast the next 14 days using only what was known that day, then check." The model was trained only on years before the test year. These numbers come from the frozen data snapshot (to 28 Aug 2026) and are kept in [`evaluation/snapshot-2026-08-28/backtest.json`](evaluation/snapshot-2026-08-28/backtest.json). The weekly job refreshes [`evaluation/backtest.json`](evaluation/backtest.json) from the latest data, so that file will move on from these.

**Average miss per forecast (donations per state, blood group and day; lower is better)**

| Test year | Same weekday last week | Recent weekday average | **Gradient-boosted trees** |
|---|---:|---:|---:|
| 2023 | 14.98 | 10.94 | **10.37** |
| 2024 | 14.80 | 10.52 | **10.01** |
| 2025 | 14.81 | 10.68 | **9.85** |
| 2026 to 28 Aug | 15.18 | 10.96 | **10.15** |

Overall the gain is modest, 5–8%. Almost all of it comes from the days the simple methods can't see coming:

| 2025 | Recent weekday average | Gradient-boosted trees | Change |
|---|---:|---:|---:|
| Public holidays | 19.7 | 13.9 | **−30%** |
| Ramadan | 12.3 | 8.7 | **−29%** |
| Other days | 9.90 | 9.86 | same |

**Do the 80% ranges hold on a year they never saw?** Each range is built on year Y−1 and checked on year Y.

| Checked on | Days inside the range | Week totals inside the range | Worst / best state |
|---|---:|---:|---:|
| 2024 | 80.3% | 79.3% | 76.8% / 85.9% |
| 2025 | 79.6% | 79.6% | 74.1% / 83.7% |
| 2026 to 28 Aug | 79.9% | 80.0% | 76.4% / 85.5% |

**The low-week warning**, scored on years it never saw. It counts every 7-day window that one forecast run covers.

| Checked on | Weeks that were really low | Warnings | Warnings that were right | Low weeks warned about |
|---|---:|---:|---:|---:|
| 2024 | 17.6% | 2,340 | 50% | 18% |
| 2025 | 23.4% | 2,839 | 69% | 23% |
| 2026 to 28 Aug | 22.4% | 2,773 | 73% | 38% |

The stricter "Likely low" level was right 89%, 100% and 95% of the time. But it fires rarely and misses most low weeks.

The recent weekday average **never** raised a warning. An average over three years can't see a holiday week coming, so a warning needs the calendar.

## How it works, step by step

### 1. Data
The official CSV ([data.gov.my, CC BY 4.0](https://data.gov.my/data-catalogue/blood_donations_state)) is downloaded once a day: daily donations by state and blood group since 2006. It is checked before use (duplicates, negative counts, gaps), and it stops rather than guesses. [`DATA_NOTES.md`](DATA_NOTES.md) lists everything odd in it and what was done about each one.

### 2. Calendar signals
For each state and day: public holidays (state by state, so Hari Hol Pahang only counts in Pahang), festive season (Hari Raya Puasa, Hari Raya Haji, Chinese New Year, Deepavali, Christmas, Gawai, Kaamatan), days to and since the nearest holiday and Hari Raya, and the day of Ramadan.

### 3. Three forecasters, compared on the same backtests
| Method | What it does |
|---|---|
| Same weekday last week | The latest known count for that weekday. The bar to beat. |
| Recent weekday average | The average for that weekday over the last 156 weeks. The Klang Valley dashboard's best method. |
| **Gradient-boosted trees** (LightGBM, Poisson loss) | One model for all states and groups. Inputs: the weekday averages above, recent levels, last year's level, and the calendar signals. |

Every input for a forecast made on day *o* uses counts up to day *o* only. A test scrambles every later count and checks that no input changes. The gradient-boosted model's settings were not tuned on any test year, and it has the lowest error in every year checked, starting with 2023.

### 4. Honest ranges (split conformal)
For each state, blood group and look-ahead (days 1–7 or 8–14), the backtest errors e = actual − forecast from the latest complete year are sorted. With α = 0.2:

```
low  = forecast + e(lo),  lo = floor((n + 1) × α/2)
high = forecast + e(hi),  hi = ceil((n + 1) × (1 − α/2))
```

Counts can't be negative, so `low` is cut at 0. The same is done for week totals.

### 5. The low-week rule (written down once and never changed)
A state and blood group's **low line** is the 30th percentile of every 7-day total in the three calendar years before the forecast's year. For the next 7 days, and for the 7 days after:
- **Likely low**: even the high end of the week total's 80% range is below the line.
- **Watch**: the forecast itself is below the line.

### 6. The scorecard
Each run saves `forecasts/<data end date>.csv` (every day, all three methods, with ranges) and `forecasts/<date>-weeks.csv` (week totals and warnings). A forecast file is never overwritten. `scorecard/scorecard.json` compares every saved forecast with the real counts once they arrive. The git history of `forecasts/` shows exactly when each forecast was published.

### 7. The bulletin
`latest/bulletin.md` (written by the first daily run) is three or four plain sentences in English and Bahasa Malaysia. They are written by fixed templates from the forecast numbers, so a sentence can never contain a number that isn't in the forecast file.

---

## Run it yourself

```bash
pip install -r requirements.txt
python -m darahwatch.evaluate --data snapshot   # about 3 minutes: backtests, ranges, warning scores
python -m darahwatch.run --data snapshot        # about 1 minute: forecast, scorecard, bulletin
python -m pytest -q
```

`--data live` downloads today's official file; `--data snapshot` uses the frozen copy in `data/` (to 28 Aug 2026, checksum recorded in the code).

## Automation
| Workflow | When | What |
|---|---|---|
| `daily.yml` | 05:37 Malaysia time | Forecast, scorecard and bulletin from the newest data, committed to `main` |
| `evaluate.yml` | Sundays | Rebuilds the backtests and the ranges the daily run uses |
| `tests.yml` | Every push and pull request | Tests, plus a full run on the snapshot |

## Data and licence
Data: Government of Malaysia, *Daily Blood Donations by Blood Group & State*, [data.gov.my](https://data.gov.my/data-catalogue/blood_donations_state), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Public holidays: the [`holidays`](https://github.com/vacanza/holidays) package. Islamic calendar: [`hijridate`](https://github.com/dralshehri/hijridate). Code: MIT licence.
