"""As-of-today regime view: trailing scored months + live-month projection.

Reads data/monthly_features.parquet + data/seasonality_by_month.csv (no MT5;
rerun 1-download.py and then 2-seasonality.py to refresh). Calendar completion
and the N>=15 validity gate are separate. The projection for the live month is
the calendar-table row (shrunk P(Trend) + 95% CI), with no within-month data.

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

    rows = feat.filter(pl.col("is_complete") & (pl.col("n") >= MIN_N)).to_dicts()
    if not rows:
        raise SystemExit("no complete months in features")
    latest = feat.to_dicts()[-1]
    if latest["is_complete"]:
        py, pm = (
            (latest["year"], latest["month"] + 1)
            if latest["month"] < 12
            else (latest["year"] + 1, 1)
        )
    else:
        py, pm = latest["year"], latest["month"]
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
