# AGENTS.md — instructions for coding agents in this repo

Read [`README.md`](README.md) for what the project is and
[`FORMULA.md`](FORMULA.md) for the math. This file is operating procedure.

## Run things

- `uv run <script>` — never bare `python` (venv is `.venv`, Python 3.14).
- MT5 scripts (`1-download.py`) need a running, logged-in MT5 terminal.
  Everything else is offline from parquet. Windows-only (MetaTrader5).
- Headless plots: `$env:MPLBACKEND="Agg"` before `uv run`, else
  `plt.show()` warns/blocks. Interactive use keeps default backend.
- Numbered scripts run in dependency order 1→2→4/5/6/7/8/9 (3 validates,
  `tmp.py` is throwaway scratch — currently the 2025 single-year chart).
  `8-forecast-check.py` replaced the deleted XGB nowcast experiment.
- Verify by execution: after touching logic, rerun the affected script
  plus `3-validate.py` / `7-formula-check.py` and quote their PASS lines.
  Never claim numbers you didn't just run.
- Forecast changes also require `8-forecast-check.py`; changes to selection or
  replay require rerunning `6-backtest.py` and `9-replay.py` so committed CSVs
  and charts agree with the code.

## Rigor rules (earned the hard way here)

- Real bars only for features: `is_filled=False`. Filled rows are flat
  placeholders (prev close, vol 0), never observations.
- N>=15 gate for GMM fit AND calendar means (`MIN_N_FIT`). Partial edge
  months score degenerate values — keep them out of every average.
- No lookahead: any claim about prediction skill must come from
  train-strictly-before-test schemes (`6-backtest.py`, `9-replay.py`),
  realized labels from the pre-window model, never the full-data one.
- The official forecast is frozen after the last D1 bar of month m for month
  m+1. A challenger replaces calendar only when calendar is excluded from the
  95% year-block Model Confidence Set and regime diagnostics pass.
- Report uncertainty with every probability (CI/n). A point estimate
  without its interval is a bug in this repo.
- Determinism: fixed seeds everywhere (`random_state=0`, seeded synth).
  Reruns must reproduce committed numbers exactly.

## Code conventions

- `uvx ruff check .` and `uvx ruff format --check .` must pass with zero
  warnings before commit. Fix by editing, not `--no-verify`.
- Conventional commits (`feat/fix/style/test/chore`), one logical change
  each. Amend only unpushed commits — amending pushed history caused a
  real divergence here (see merge `758bb40`).
- Shared math lives in `2-seasonality.py` (`month_features`,
  `geomean_row`, `month_rows`) — import it via importlib (filenames start
  with digits), don't duplicate it.
- Shared forecasting, bootstrap intervals, candidate fitting, and Model
  Confidence Set selection live in `forecasting.py`; keep scripts 5/6/8/9 as
  thin consumers of that module.
- Matplotlib house style: no grid (`rcParams["axes.grid"] = False`),
  gray dashed month separators incl. first/last bar, tight x-limits
  (`set_xlim(first, last)`), UTC dates.

## Windows/pwsh pitfalls hit in this repo

- `python -c "..."` with nested quotes breaks in pwsh — write a temp
  `_*.py` file, `uv run` it, delete it. Never commit `_*.py`.
- Don't `print(df)` on polars frames: it crashes cp1252 consoles (unicode box
  chars) — print `.to_dicts()` rows instead. `tzdata` is a required dep
  for any tz-aware polars work on Windows.
- `except A, B:` parses as a tuple catch — always parenthesize anyway.
- `git status --porcelain` (not `git diff --stat | Select-Object`) for
  reliable state; `Select-Object -First/Last N` instead of `head`.
