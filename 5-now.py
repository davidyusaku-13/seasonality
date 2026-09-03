"""As-of-today regime view: trailing scored months + next-month projection.

Reads data/monthly_features.parquet + data/seasonality_by_month.csv (no MT5,
so rerun 1-download.py first to refresh). A month counts as complete when
N>=15 (same bar as the GMM fit). The projection for the upcoming month is
the calendar-table row (shrunk P(Trend) + 95% CI); the XGB partial-month
model (models/xauusd_partial.json) nowcasts any in-progress month.

Output: charts/now.png
"""

import importlib.util
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import xgboost as xgb

matplotlib.rcParams["axes.grid"] = False

SPEC = Path(__file__).with_name("8-xgb.py")
_spec = importlib.util.spec_from_file_location("xgb8", SPEC)
xgb8 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(xgb8)

SRC = Path("data/xauusd_d1.parquet")
FEAT = Path("data/monthly_features.parquet")
TAB = Path("data/seasonality_by_month.csv")
MODEL = Path("models/xauusd_partial.json")
METRICS = Path("models/xgb_partial_metrics.json")
OUT = Path("charts/now.png")
MIN_N = 15
TRAIL = 12


def bucket_of(k: int) -> str:
    for lo, hi in ((3, 7), (8, 12), (13, 17), (18, 20)):
        if lo <= k <= hi:
            return f"{lo}-{hi}"
    return "overall"


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

    nowcasts: dict[str, tuple[float, float]] = {}
    try:
        model = xgb.XGBRegressor()
        model.load_model(MODEL)
        metrics = json.loads(METRICS.read_text())
        real = pl.read_parquet(SRC).filter(~pl.col("is_filled")).sort("time")
        for mo in xgb8.month_rows(real):
            if mo["ym"] in {r["ym"] for r in pending}:
                f = np.array(
                    xgb8.build_partial_features(
                        mo["h"], mo["low"], mo["c"], mo["c0"], len(mo["c"])
                    )
                ).reshape(1, -1)
                bkt = bucket_of(len(mo["c"]))
                mae = metrics["mae_by_k_bucket"].get(bkt, metrics["test_mae"])
                nowcasts[mo["ym"]] = (float(model.predict(f)[0]), float(mae))
                print(
                    f"nowcast {mo['ym']} (k={len(mo['c'])}): "
                    f"S_m~{nowcasts[mo['ym']][0]:.3f} (+/-{mae:.3f} test MAE)"
                )
    except FileNotFoundError, OSError:
        print("nowcast model unavailable (run 8-xgb.py first)")

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
    for ym, (val, mae) in nowcasts.items():
        if ym == f"{py}-{pm:02d}":
            ax.errorbar(
                [xs[-1]],
                [val],
                yerr=[[mae], [mae]],
                fmt="D",
                color="orange",
                ecolor="orange",
                capsize=5,
                label="nowcast",
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


if __name__ == "__main__":
    main()
