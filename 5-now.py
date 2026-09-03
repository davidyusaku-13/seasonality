"""As-of-today regime view: trailing scored months + next-month projection.

Reads data/monthly_features.parquet + data/seasonality_by_month.csv (no MT5,
so rerun 1-download.py first to refresh). A month counts as complete when
N>=15 (same bar as the GMM fit). The projection for the upcoming month is
the calendar-table row (shrunk P(Trend) + 95% CI): a real projection with
no within-month (k>0) data.

Output: charts/now.png
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import polars as pl

matplotlib.rcParams["axes.grid"] = False

FEAT = Path("data/monthly_features.parquet")
TAB = Path("data/seasonality_by_month.csv")
OUT = Path("charts/now.png")
MIN_N = 15
TRAIL = 12


def main() -> None:
    feat = pl.read_parquet(FEAT).sort(["year", "month"])
    tab = pl.read_csv(TAB).sort("month")

    rows = feat.to_dicts()
    pending = []
    while rows and rows[-1]["n"] < MIN_N:
        pending.append(rows.pop())
    if not rows:
        raise SystemExit("no complete months in features")
    last = rows[-1]
    py, pm = (
        (last["year"], last["month"] + 1)
        if last["month"] < 12
        else (last["year"] + 1, 1)
    )
    proj = tab.filter(pl.col("month") == pm).to_dicts()[0]
    trail = rows[-TRAIL:]
    asof = feat["end"].max()

    print(f"as of {asof} (data end)")
    print(f"{'month':<8} {'S_m':>6} {'q':>6}")
    for r in trail:
        print(f"{r['ym']:<8} {r['s']:>6.3f} {r['q_trend']:>6.3f}")
    print(
        f"projection {py}-{pm:02d} ({proj['month_name']}): "
        f"P(Trend)={proj['p_trend']:.1%} "
        f"95% CI [{proj['ci95_lo']:.1%},{proj['ci95_hi']:.1%}] "
        f"(n={proj['n']})"
    )

    xs = list(range(len(trail) + 1))
    labels = [r["ym"] for r in trail] + [f"{py}-{pm:02d}*"]
    qs = [r["q_trend"] for r in trail]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(xs[:-1], qs, marker="o", linewidth=2, label="q_m actual")
    ax.errorbar(
        [xs[-1]],
        [proj["p_trend"]],
        yerr=[
            [proj["p_trend"] - proj["ci95_lo"]],
            [proj["ci95_hi"] - proj["p_trend"]],
        ],
        fmt="s",
        color="green",
        ecolor="black",
        capsize=5,
        label="projection",
    )
    ax.axvline(xs[-1] - 0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylim(0, 1)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("0=ranging 1=trend")
    ax.set_title(
        f"XAUUSD regime now (as of {asof}): trailing {len(trail)}m + {proj['month_name']} projection"
    )
    ax.legend()
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"saved={OUT.resolve()}")


if __name__ == "__main__":
    main()
