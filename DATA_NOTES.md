# Data notes

Everything odd found in the official data, and what DarahWatch does about it. The checks run on
every download (`darahwatch/data.py`, `audit()` and `series()`).

## The file
- One row per date, state and blood group: `A`, `B`, `AB`, `O` and `all`. Snapshot in `data/`: 490,425 rows,
  1 Jan 2006 to 28 Aug 2026, 13 places: 12 states and W.P. Kuala Lumpur.
- No missing values, no duplicate keys, no negative counts in the snapshot. If a future download has
  duplicates or negative counts, the run **stops** instead of guessing.

## Decisions

| What | Why it matters | What DarahWatch does |
|---|---|---|
| **12 states and Kuala Lumpur, not all 16 states and territories.** Perlis, Labuan and Putrajaya have no rows of their own. | The data comes from the Ministry of Health's main collection sites, not every site in the country. W.P. Kuala Lumpur includes Putrajaya and mobile drives around Selangor. | Forecasts the 13 places that exist and says so. Never invents the others. |
| **`all` is sometimes 1 to 3 more than A + B + AB + O** (280 state-days, 197 of them in Kuala Lumpur, increasing over the years). | Most likely donations recorded without a blood group. | Uses the four groups, never `all`, so nothing is counted twice. Totals may be up to 3 lower than the official `all` on those days. |
| **Weekends differ by state, and they changed.** Some states rest on Friday and Saturday (Kelantan, for example). Johor did too from 2014 to 2024, then moved back to Saturday and Sunday on 1 Jan 2025 ([The Star](https://www.thestar.com.my/news/nation/2024/10/07/johor039s-weekend-to-be-saturday-sunday-from-jan-1-2025)). Johor's average Friday went from 29 donations in 2024 to 55 in 2025 and 72 in 2026. | A weekday average over three years lags behind a weekend change. | No weekend calendar is hard-coded, because an error in it would be silent. Instead, the model sees the same weekday over the last 1, 4, 13, 52 and 156 weeks, so a change shows up in the recent averages within weeks. |
| **Kedah's level fell by about two-thirds in 2020 and never came back.** | Probably a change in which sites report (COVID-19 began then too). Any long-run average for Kedah is wrong after 2020. | Recent levels (7, 28 and 91 days) are model inputs, and the low-week line uses only the last three years. |
| **Public holidays and Ramadan** move collections a lot. Hari Raya weeks are among the lowest of the year. | The two simple methods can't know a holiday is coming. | State-by-state holidays from the `holidays` package (Malaysia, with each state's own days). Ramadan runs from 1 Ramadan (Umm al-Qura calendar) to the day before Hari Raya Puasa as Malaysia observed it, and is never longer than 30 days. |
| **Days with zero donations** (for example AB on some days in small states). | Real, not missing. | Kept as zero. The model uses a Poisson loss, which handles counts and zeros. Ranges are cut at zero. |
| **The newest days can be incomplete.** | A state that hasn't reported yet would look like a collapse. | Analyses only up to the newest date on which every place reports all four groups. |

## Not used, and why
- **School holidays**: no official machine-readable calendar covers 2006 onwards. Left out rather than typed by hand.
- **Weather and blood drives**: no free daily source by state.
