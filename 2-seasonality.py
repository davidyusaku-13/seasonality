"""Descriptive monthly regime features for XAUUSD D1 (FORMULA.md section 3).

Pipeline:
  data/xauusd_d1.parquet (real bars only, is_filled=False)
    -> per calendar month features [T_range, T_direction, T_mono] + S_m
    -> 2-component GMM -> q_m = P(Trend | X_m)
    -> group by month-of-year -> descriptive P(Trend | Jan..Dec) + CI

Outputs:
  data/monthly_features.parquet  (one row per year-month, with q_trend)
  data/seasonality_by_month.csv  (final P(Trend)/P(Sideways) table)
  seasonality.png                (bar chart with 95% CI + sample counts)
  monthly_scores.png             (q_m and S_m through time)

Mixture fit uses completed, valid months only (N>=15) so partial edge months
(2007-06, the live month) don't distort components; q is predicted where valid.
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import kendalltau
from sklearn.mixture import GaussianMixture

SRC = Path("data/xauusd_d1.parquet")
FEAT_OUT = Path("data/monthly_features.parquet")
TAB_OUT = Path("data/seasonality_by_month.csv")
FIG_SEASON = Path("seasonality.png")
FIG_TIME = Path("monthly_scores.png")
MIN_N_FIT = 15
BOUND_TOL = 1e-10

MONTH_NAMES = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]


def clip_unit(value: float, label: str) -> float:
    """Clip floating-point noise, but reject material invariant violations."""
    if not np.isfinite(value) or value < -BOUND_TOL or value > 1 + BOUND_TOL:
        raise ValueError(f"{label} outside [0,1]: {value}")
    return float(np.clip(value, 0, 1))


def month_features(h: np.ndarray, low: np.ndarray, c: np.ndarray, c0: float):
    """FORMULA section 3. Returns (T_range, T_direction, T_mono, S_m)."""
    n = len(c)
    if n <= 1:
        return 0.0, 0.0, 0.0, 0.0
    c_prev = np.concatenate([[c0], c[:-1]])
    tr = np.maximum(h, c_prev) - np.minimum(low, c_prev)
    rm = float(np.max(np.concatenate([[c0], h])) - np.min(np.concatenate([[c0], low])))
    if rm <= 0 or tr.sum() <= 0:
        return (float("nan"),) * 4
    t_range = 1.0 - float(np.log(tr.sum() / rm) / np.log(n))
    t_range = clip_unit(t_range, "T_range")

    r = np.log(c / c_prev)
    denom = n * float(np.sum(r * r))
    t_dir = float(abs(r.sum()) / np.sqrt(denom)) if denom > 0 else 0.0
    t_dir = clip_unit(t_dir, "T_direction")

    p = np.concatenate([[np.log(c0)], np.log(c)])
    try:
        tau, _ = kendalltau(np.arange(n + 1), p)
    except ValueError:
        tau = np.nan
    t_mono = float(abs(tau)) if tau is not None and np.isfinite(tau) else 0.0
    t_mono = clip_unit(t_mono, "T_mono")

    s = geomean_row(np.array([t_range, t_dir, t_mono]))
    return t_range, t_dir, t_mono, s


def geomean_row(t: np.ndarray) -> float:
    t = np.clip(t, 0, 1)
    return 0.0 if t.min() <= 0 else float(np.prod(t) ** (1 / 3))


def forecast_origin_features(
    h: np.ndarray, low: np.ndarray, c: np.ndarray, c0: float
) -> tuple[float, float, float, float]:
    """Price-only state features known after a month's final D1 bar."""
    if len(c) == 0 or c0 <= 0 or np.any(h <= 0) or np.any(low <= 0) or np.any(c <= 0):
        return (float("nan"),) * 4
    c_prev = np.concatenate([[c0], c[:-1]])
    r = np.log(c / c_prev)
    squared = r * r
    rv = float(squared.sum())
    range_var = float(np.square(np.log(h / low)).sum() / (4 * np.log(2)))
    if rv <= 0:
        return rv, range_var, 0.0, 0.0
    jump_share = float(squared.max() / rv)
    upside = float(squared[r >= 0].sum())
    downside = float(squared[r < 0].sum())
    semivar_imbalance = float(abs(upside - downside) / rv)
    return (
        rv,
        range_var,
        clip_unit(jump_share, "jump_share"),
        clip_unit(semivar_imbalance, "semivar_imbalance"),
    )


def month_rows(real: pl.DataFrame) -> list[dict]:
    """Per-month OHLC arrays with prior close (real bars only)."""
    times = real["time"].to_list()
    closes_all = [float(v) for v in real["close"].to_list()]
    latest = max((d.year, d.month) for d in real["date"].to_list())
    out = []
    for y, m in sorted({(d.year, d.month) for d in real["date"].to_list()}):
        mb = real.filter(
            (pl.col("date").dt.year() == y) & (pl.col("date").dt.month() == m)
        ).sort("time")
        idx = times.index(mb["time"].min())
        c0 = closes_all[idx - 1] if idx > 0 else float(mb["open"].to_list()[0])
        out.append(
            {
                "year": y,
                "month": m,
                "ym": f"{y}-{m:02d}",
                "is_complete": (y, m) < latest,
                "h": np.array(mb["high"].to_list(), dtype=float),
                "low": np.array(mb["low"].to_list(), dtype=float),
                "c": np.array(mb["close"].to_list(), dtype=float),
                "c0": float(c0),
            }
        )
    return out


def build_monthly(real: pl.DataFrame) -> pl.DataFrame:
    times = real["time"].to_list()
    closes_all = real["close"].to_list()
    yms = sorted({(d.year, d.month) for d in real["date"].to_list()})
    latest = max(yms)
    rows = []
    for y, m in yms:
        mb = real.filter(
            (pl.col("date").dt.year() == y) & (pl.col("date").dt.month() == m)
        ).sort("time")
        first_time = mb["time"].min()
        idx = times.index(first_time)
        c0 = closes_all[idx - 1] if idx > 0 else float(mb["open"].to_list()[0])
        h = mb["high"].to_numpy()
        low = mb["low"].to_numpy()
        c = mb["close"].to_numpy()
        tr, td, tm, s = month_features(h, low, c, float(c0))
        rv, range_var, jump_share, semivar_imbalance = forecast_origin_features(
            h, low, c, float(c0)
        )
        rows.append(
            {
                "year": y,
                "month": m,
                "ym": f"{y}-{m:02d}",
                "is_complete": (y, m) < latest,
                "start": mb["date"].min(),
                "end": mb["date"].max(),
                "n": mb.height,
                "c0": float(c0),
                "cn": float(c[-1]),
                "ret_total": float(np.log(c[-1] / c0)),
                "t_range": tr,
                "t_direction": td,
                "t_mono": tm,
                "s": s,
                "realized_variance": rv,
                "range_variance": range_var,
                "jump_share": jump_share,
                "semivar_imbalance": semivar_imbalance,
            }
        )
    return pl.DataFrame(rows).sort(["year", "month"])


def fit_mixture(feat: pl.DataFrame) -> np.ndarray:
    X = feat.select(["t_range", "t_direction", "t_mono"]).to_numpy()
    valid_mask = np.isfinite(X).all(axis=1)
    fit_mask = (feat["is_complete"] & (feat["n"] >= MIN_N_FIT)).to_numpy() & valid_mask
    gmm = GaussianMixture(
        n_components=2,
        covariance_type="full",
        n_init=10,
        random_state=0,
        reg_covar=1e-3,
    )
    gmm.fit(X[fit_mask])
    order = np.argsort(gmm.means_.mean(axis=1))
    trend_idx = int(order[1])  # component with larger mean values = Trend
    q = np.full(len(X), np.nan)
    q[valid_mask] = gmm.predict_proba(X[valid_mask])[:, trend_idx]
    print(
        f"gmm fit on {int(fit_mask.sum())}/{len(X)} months (N>={MIN_N_FIT}); "
        f"trend_comp={trend_idx} means={gmm.means_.round(3).tolist()} "
        f"weights={gmm.weights_.round(3).tolist()}"
    )
    return q


def calendar_table(feat: pl.DataFrame) -> pl.DataFrame:
    full = feat.filter(
        pl.col("is_complete")
        & (pl.col("n") >= MIN_N_FIT)
        & pl.col("q_trend").is_finite()
    )
    qv = full["q_trend"].to_numpy()
    months = full["month"].to_numpy()
    rows = []
    for m in range(1, 13):
        qq = qv[months == m]
        n = len(qq)
        raw = float(qq.mean()) if n else float("nan")
        shrunk = float((0.5 + qq.sum()) / (n + 1)) if n else float("nan")
        std = float(qq.std(ddof=1)) if n > 1 else 0.0
        se = std / np.sqrt(n) if n else 0.0
        rows.append(
            {
                "month": m,
                "month_name": MONTH_NAMES[m - 1],
                "n": n,
                "p_trend_raw": raw,
                "p_trend": shrunk,
                "p_sideways": 1 - shrunk,
                "std": std,
                "ci95_lo": max(0.0, shrunk - 1.96 * se),
                "ci95_hi": min(1.0, shrunk + 1.96 * se),
                "q_min": float(qq.min()) if n else float("nan"),
                "q_max": float(qq.max()) if n else float("nan"),
            }
        )
    return pl.DataFrame(rows).sort("month")


def expected_brier(p: np.ndarray, q: np.ndarray) -> float:
    """Expected binary Brier loss when q is the posterior event probability."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return float(np.mean((p - q) ** 2 + q * (1 - q)))


def main() -> None:
    matplotlib.rcParams["axes.grid"] = False
    real = pl.read_parquet(SRC).filter(~pl.col("is_filled")).sort("time")
    print(f"loaded {real.height} real bars {real['date'].min()}..{real['date'].max()}")

    feat = build_monthly(real)
    print(f"months: {feat.height} ({feat['ym'].min()}..{feat['ym'].max()})")
    q = fit_mixture(feat)
    feat = feat.with_columns(pl.Series("q_trend", q))
    FEAT_OUT.parent.mkdir(parents=True, exist_ok=True)
    feat.write_parquet(FEAT_OUT)
    print(f"wrote {FEAT_OUT} rows={feat.height}")

    tab = calendar_table(feat)
    tab.write_csv(TAB_OUT)
    print(f"wrote {TAB_OUT}")
    print(f"{'mon':<5} {'n':>3} {'P(tr)':>7} {'P(sw)':>7} {'raw':>7} {'95% CI':>15}")
    print("-" * 52)
    for r in tab.to_dicts():
        print(
            f"{r['month_name']:<5} {r['n']:>3} {r['p_trend']:>7.1%} {r['p_sideways']:>7.1%} "
            f"{r['p_trend_raw']:>7.1%} [{r['ci95_lo']:>5.1%},{r['ci95_hi']:>5.1%}]"
        )

    # fig 1: seasonality bars with CI
    fig, ax = plt.subplots(figsize=(12, 5))
    xs = tab["month"].to_list()
    ax.bar(xs, tab["p_trend"].to_list(), width=0.7, label="P(Trend) shrunk")
    ax.errorbar(
        xs,
        tab["p_trend"].to_list(),
        yerr=[
            (np.array(tab["p_trend"]) - np.array(tab["ci95_lo"])),
            (np.array(tab["ci95_hi"]) - np.array(tab["p_trend"])),
        ],
        fmt="none",
        ecolor="black",
        capsize=4,
    )
    ax.plot(
        xs, tab["p_trend_raw"].to_list(), marker="o", linewidth=1, label="raw mean q"
    )
    for r in tab.to_dicts():
        ax.text(r["month"], r["p_trend"] + 0.02, f"n={r['n']}", ha="center", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_xticks(xs)
    ax.set_xticklabels(tab["month_name"].to_list())
    ax.set_ylabel("P(Trend)")
    ax.set_title("XAUUSD historical regime seasonality: P(Trend | calendar month)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_SEASON, dpi=150)
    print(f"saved={FIG_SEASON.resolve()}")

    # fig 2: q_m and S_m through time
    fig2, ax2 = plt.subplots(figsize=(14, 4))
    ax2.plot(
        feat["ym"].to_list(),
        feat["q_trend"].to_list(),
        linewidth=1,
        label="q=P(Trend|X)",
    )
    ax2.plot(
        feat["ym"].to_list(), feat["s"].to_list(), linewidth=1, alpha=0.7, label="S_m"
    )
    ax2.set_ylim(0, 1)
    step = max(1, feat.height // 20)
    ax2.set_xticks(range(0, feat.height, step))
    ax2.set_xticklabels(
        [feat["ym"][i] for i in range(0, feat.height, step)], rotation=30, ha="right"
    )
    ax2.set_ylabel("0=ranging 1=trend")
    ax2.set_title("Descriptive monthly latent trend probability through time")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(FIG_TIME, dpi=150)
    print(f"saved={FIG_TIME.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()
