# XAUUSD Next-Month Regime Forecast

This project estimates whether the **next XAUUSD calendar month** will trend
persistently or trade sideways/choppily. It is direction-agnostic: Trend can be
bullish or bearish. It does not issue a trading signal.

Formula and statistical contract: [`FORMULA.md`](FORMULA.md). Live view:
[`charts/now.png`](charts/now.png).

## Current forecast

After the final D1 bar of August 2026, the frozen forecast for September 2026
is:

| Target | P(Trend) | P(Sideways) | 95% bootstrap interval | Model | Training transitions |
|---|---:|---:|---:|---|---:|
| 2026-09 | 61.2% | 38.8% | [39.1%, 80.4%] | calendar fallback | 181 |

No challenger currently earns promotion. Across 145 strictly out-of-sample
monthly forecasts, partially pooled month effects give the best Brier point
estimate, but calendar remains inside the 95% Model Confidence Set:

| Model | Brier | Log loss | Brier difference vs calendar, 95% CI |
|---|---:|---:|---:|
| Calendar | 0.2569 | 0.7163 | 0.0000 [0.0000, 0.0000] |
| Calibrated calendar | 0.2501 | 0.6940 | -0.0067 [-0.0247, +0.0111] |
| Seasonal soft Markov | 0.2501 | 0.6933 | -0.0068 [-0.0271, +0.0131] |
| Regularized logistic | 0.2494 | 0.6921 | -0.0074 [-0.0286, +0.0139] |
| Cyclic seasonality | 0.2489 | 0.6937 | -0.0080 [-0.0269, +0.0106] |
| Partially pooled months | **0.2484** | 0.6951 | -0.0085 [-0.0239, +0.0066] |
| Multiscale price state | 0.2534 | 0.6999 | -0.0035 [-0.0297, +0.0235] |

Pooling does reduce September's sampling interval: the partially pooled model
estimates 53.6% `[44.8%, 61.1%]`, a 16.2-point interval versus calendar's
41.3-point interval. This is a more precise but less decisive estimate near
50%, and the evidence is not strong enough to replace the official baseline.
The multiscale price features did not improve the pooled models.

Intervals become narrower by sharing information across adjacent calendar
months and shrinking sparse month effects toward the global rate. They cannot
be honestly narrowed just by changing the confidence level or ignoring target
and model-selection uncertainty; more independent history or genuinely useful
predictors are still needed for greater certainty.

The correct interpretation is therefore “calendar lean, not validated
transition signal.” The realized outcome is itself a prior-only GMM latent
state, not independently observed ground truth.

## Method

Completed D1 months receive three price-behavior scores in `[0,1]`: range
efficiency, directional coherence, and temporal monotonicity. A two-component
GMM fitted strictly on earlier months assigns a soft realized trend state
`q_t`. Candidate models then forecast `q_(t+1)` from pooled calendar seasonality,
recent states, and multiscale D1 volatility/jump measurements. Monthly
expanding-window evaluation and a year-block Model Confidence Set decide
whether any challenger may replace the calendar baseline.

Only complete months with at least 15 real bars enter fitting or averages.
Filled holiday placeholders are never observations. Every probability is
reported with uncertainty and sample size.

## Pipeline

Run with `uv run <script>` in dependency order. Only data download requires a
running, logged-in MetaTrader 5 terminal.

| # | Script | Purpose |
|---|---|---|
| 1 | `1-download.py` | Download and validate XAUUSD D1 bars |
| 2 | `2-seasonality.py` | Build monthly features and descriptive full-history regimes |
| 3 | `3-validate.py` | Validate economic behavior on synthetic paths |
| 4 | `4-charts.py` | Draw descriptive yearly price/regime charts |
| 5 | `5-now.py` | Select the eligible model and publish the next-month forecast |
| 6 | `6-backtest.py` | Run monthly expanding-window comparison and promotion test |
| 7 | `7-formula-check.py` | Check formula invariants and real-data diagnostics |
| 8 | `8-forecast-check.py` | Check causality, determinism, transitions, and forecast output |
| 9 | `9-replay.py` | Reconstruct 2026 forecasts and model selection as known then |

Primary machine-readable outputs:

- `data/next_month_forecast.csv`
- `data/next_month_candidates.csv`
- `data/forecast_backtest.csv`
- `data/model_comparison.csv`
- `data/replay_2026.csv`

To reproduce the forecast and its checks after the parquet data is available:

```powershell
$env:MPLBACKEND="Agg"
uv run 2-seasonality.py
uv run 6-backtest.py
uv run 5-now.py
uv run 9-replay.py
uv run 3-validate.py
uv run 7-formula-check.py
uv run 8-forecast-check.py
uvx ruff check .
uvx ruff format --check .
```

## Current validation

- Formula and synthetic regime checks pass.
- Prior-only targets are deterministic; altering a future month cannot alter
  earlier targets.
- All 145 evaluation folds have ordered component means and favor two GMM
  components by BIC.
- Synthetic checks verify cyclic/month pooling, multiscale causality, and Model
  Confidence Set inclusion and exclusion behavior.
- Blind 2026 replay: 8 forecasts, Brier 0.2101, hard agreement 6/8. Calendar
  remained the official model at every historical origin.

All dates are UTC and all stochastic procedures use fixed seeds.
