"""XAUUSD 2025 seasonality (FORMULA.md steps 1-8).

Top: daily close with month separators. Bottom: monthly S_m in 0-1
(0=ranging, 1=trend) + its 3 components. Uses real bars only.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import kendalltau

SRC = Path("data/xauusd_d1.parquet")
OUT = Path("xauusd_2025.png")


def month_features(h: np.ndarray, low: np.ndarray, c: np.ndarray, c0: float):
    n = len(c)
    if n <= 1:
        return 0.0, 0.0, 0.0, 0.0
    c_prev = np.concatenate([[c0], c[:-1]])
    tr = np.maximum(h, c_prev) - np.minimum(low, c_prev)
    rm = float(np.max(np.concatenate([[c0], h])) - np.min(np.concatenate([[c0], low])))
    if rm <= 0 or tr.sum() <= 0:
        t_range = 0.0
    else:
        t_range = 1.0 - float(np.log(tr.sum() / rm) / np.log(n))
    t_range = float(np.clip(t_range, 0, 1))

    r = np.log(c / c_prev)
    denom = n * float(np.sum(r * r))
    t_dir = float(abs(r.sum()) / np.sqrt(denom)) if denom > 0 else 0.0
    t_dir = float(np.clip(t_dir, 0, 1))

    p = np.concatenate([[np.log(c0)], np.log(c)])
    t = np.arange(n + 1)
    try:
        tau, _ = kendalltau(t, p)
    except Exception:
        tau = np.nan
    t_mono = float(abs(tau)) if tau is not None and np.isfinite(tau) else 0.0
    t_mono = float(np.clip(t_mono, 0, 1))

    s = float((t_range * t_dir * t_mono) ** (1 / 3)) if min(t_range, t_dir, t_mono) > 0 else 0.0
    return t_range, t_dir, t_mono, s


def main() -> None:
    real = pl.read_parquet(SRC).filter(~pl.col("is_filled")).sort("time")
    y2025 = real.filter(pl.col("date").dt.year() == 2025).sort("time")
    if y2025.height == 0:
        raise SystemExit(f"no 2025 rows in {SRC}")

    dates = y2025["date"].to_list()
    closes = y2025["close"].to_list()

    # per-month features; C0 = last real close before the month
    months, rows = [], []
    for m in range(1, 13):
        mb = y2025.filter(pl.col("date").dt.month() == m).sort("time")
        if mb.height == 0:
            continue
        first_time = mb["time"].min()
        prev = real.filter(pl.col("time") < first_time).sort("time")
        c0 = float(prev["close"].to_list()[-1])
        h = mb["high"].to_numpy()
        low = mb["low"].to_numpy()
        c = mb["close"].to_numpy()
        tr, td, tm, s = month_features(h, low, c, c0)
        months.append(m)
        rows.append((m, mb.height, tr, td, tm, s))

    print(f"{'mon':<5} {'N':>3} {'T_range':>8} {'T_dir':>7} {'T_mono':>7} {'S_m':>6}")
    print("-" * 44)
    for m, n, tr, td, tm, s in rows:
        print(f"{m:<5} {n:>3} {tr:>8.3f} {td:>7.3f} {tm:>7.3f} {s:>6.3f}")

    fig, (ax_p, ax_s) = plt.subplots(2, 1, figsize=(14, 8), sharex=False,
                                     gridspec_kw={"height_ratios": [3, 1]})

    # top: price + month separators (first trading day + last day)
    ax_p.plot(dates, closes, linewidth=1.2, label="close")
    seen: dict = {}
    for d in dates:
        seen.setdefault(d.month, d)
    for mm in sorted(seen):
        ax_p.axvline(seen[mm], color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax_p.axvline(dates[-1], color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax_p.set_title("XAUUSD D1 2025 (vertical = month start)")
    ax_p.set_ylabel("close (USD)")
    ax_p.legend()

    # bottom: monthly seasonality 0-1
    sm = [r[5] for r in rows]
    ax_s.bar(months, sm, width=0.7, label="S_m")
    ax_s.plot(months, [r[2] for r in rows], marker="o", linewidth=1, label="T_range")
    ax_s.plot(months, [r[3] for r in rows], marker="o", linewidth=1, label="T_dir")
    ax_s.plot(months, [r[4] for r in rows], marker="o", linewidth=1, label="T_mono")
    for mm in months:
        ax_s.axvline(mm - 0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax_s.axvline(months[-1] + 0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax_s.set_ylim(0, 1)
    ax_s.set_xticks(months)
    ax_s.set_xlabel("month")
    ax_s.set_ylabel("0=ranging 1=trend")
    ax_s.set_title("Monthly seasonality S_m (geometric mean)")
    ax_s.legend(ncol=4)

    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"saved={OUT.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()
