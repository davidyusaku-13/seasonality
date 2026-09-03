# XAUUSD Monthly Regime Seasonality

Do gold months trend or chop? This repo answers that one question — **not**
which direction price goes, but *how* it moves: persistent trend vs
sideways chop, per calendar month, direction-agnostic.

Spec: [`FORMULA.md`](FORMULA.md). Live view: [`charts/now.png`](charts/now.png).

## Results (current)

`P(Trend)` by calendar month, 19–20 Januaries/Februaries/… of history:

| Month | P(Trend) | P(Sideways) | 95% CI |
|---|---:|---:|---|
| Jan | 62% | 38% | [41%, 83%] |
| Feb | 68% | 32% | [48%, 87%] |
| Mar | 32% | 68% | [13%, 51%] |
| Apr | 26% | 74% | [8%, 43%] |
| May | 40% | 60% | [20%, 60%] |
| Jun | 65% | 35% | [45%, 84%] |
| Jul | 51% | 49% | [33%, 70%] |
| Aug | 53% | 47% | [31%, 74%] |
| Sep | 62% | 38% | [42%, 82%] |
| Oct | 42% | 58% | [22%, 63%] |
| Nov | 44% | 56% | [25%, 64%] |
| Dec | 61% | 39% | [41%, 81%] |

Honest reading: real in-sample spreads (Feb 68% vs Apr 26%), but n≈19–20
makes every CI ±20pp, and walk-forward skill vs climatology is only +0.007
(Brier) over 2012–2026 — a lean, not a signal. High-confidence `q` calls
are overconfident (predicted 0.69 → realized 0.42). See
[`data/seasonality_by_month.csv`](data/seasonality_by_month.csv).

## Method (one paragraph)

D1 bars → per calendar month three scores in [0, 1]: **range efficiency**
(did the month expand its range without wandering), **directional
coherence** (did daily moves agree *and* spread across days — a lone jump
scores `1/sqrt(N)`), **temporal monotonicity** (`|Kendall tau|` of log
close vs time — catches trend-then-reversal). Geometric mean → `S_m`.
A 2-component GMM on `[T_range, T_direction, T_mono]` gives
`q_m = P(Trend | month)`; calendar means with Jeffreys-style shrinkage
(`(0.5 + Σq) / (N+1)`) give the table above.

## Pipeline

Run in order with `uv run <script>` (Windows + MT5 terminal required
only for step 1):

| # | Script | Does | Output |
|---|---|---|---|
| 1 | `1-download.py` | Full XAUUSD D1 via MT5; flat-fills 46 missing weekdays (45 NYSE holidays + 2022-12-01 gap) | `data/xauusd_d1.parquet` (5119 rows, 2007-06-22…2026-09-03) |
| 2 | `2-seasonality.py` | Monthly features, GMM → `q`, calendar table | `data/monthly_features.parquet` (232 mo), `data/seasonality_by_month.csv`, `seasonality.png`, `monthly_scores.png` |
| 3 | `3-validate.py` | 10 synthetic cases, 12 checks | console: must print `VALIDATION: PASS` |
| 4 | `4-charts.py` | Yearly price + seasonality drawings | `charts/yearly/xauusd_YYYY.png` (20) |
| 5 | `5-now.py` | Trailing 12 scored months + next-month projection | `charts/now.png` |
| 6 | `6-backtest.py` | Walk-forward 2012–2026 skill vs climatology | console numbers |
| 7 | `7-formula-check.py` | Formula invariants, edges, null distribution | console: must print `FORMULA-CHECK: PASS` |
| 9 | `9-replay.py` | Blind Jan–Aug 2026 replay (refits on pre-month data only) | `data/replay_2026.csv` |

No `8-*.py` — it was the removed XGB nowcast (see history). `tmp.py`
is scratch (currently: 2025 single-year mirror of script 4).

## Validation status

- Formula: 11/11 PASS (`7-formula-check.py`) — exact hand values, scale
  ×100 and time-reversal invariance to 1e-9, noise monotonicity, no NaN
  over 230 full months, `corr(S,N) = −0.09`, GBM null mean `S = 0.34`.
- Regimes: 12/12 PASS (`3-validate.py`), incl. single-shock
  `T_dir == 1/sqrt(N)` and bull/bear symmetry.
- Replay Jan–Aug 2026, zero lookahead: Brier 0.180 vs 0.207
  (skill +0.128), hard calls 6/8, refit stability PASS (max drift 0.004).
- Live (as of 2026-09-03): Sep 2026 in progress (N=3, unscored);
  projection Sep = 62% [42%, 82%].

## Data notes

- Single consistent D1 feed (Vantage Markets live). Broker Sunday bars
  kept as-is; features use real bars only (`is_filled=False`).
- Months with N<15 are scored but excluded from GMM fit and calendar
  means (currently: 2007-06, 2026-09).
- All dates UTC. Deterministic: fixed seeds; reruns reproduce every
  number above.
