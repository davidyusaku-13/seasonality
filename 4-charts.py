"""Yearly seasonality drawings: calendar-month S_m + q_m as the main chart.

One PNG per year in charts/yearly/xauusd_YYYY.png.
Single 0-1 panel: S_m bars, q_m=P(Trend|X) line, T components thin.
Features come from data/monthly_features.parquet (no recompute).
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import polars as pl

matplotlib.rcParams["axes.grid"] = False

FEAT = Path("data/monthly_features.parquet")
OUT_DIR = Path("charts/yearly")


def draw_year(year: int, feat: pl.DataFrame) -> Path:
    yf = feat.filter(pl.col("year") == year).sort("month")
    if yf.height == 0:
        raise SystemExit(f"no months for {year}")
    months = yf["month"].to_list()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(months, yf["s"].to_list(), width=0.7, alpha=0.85, label="S_m")
    ax.plot(months, yf["q_trend"].to_list(), marker="o", linewidth=2, color="black", label="q_m")
    ax.plot(months, yf["t_range"].to_list(), marker=".", linewidth=1, alpha=0.7, label="T_range")
    ax.plot(months, yf["t_direction"].to_list(), marker=".", linewidth=1, alpha=0.7, label="T_dir")
    ax.plot(months, yf["t_mono"].to_list(), marker=".", linewidth=1, alpha=0.7, label="T_mono")
    for mm in months:
        ax.axvline(mm - 0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axvline(months[-1] + 0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_ylim(0, 1)
    ax.set_xlim(months[0] - 0.5, 12.5)
    ax.set_xticks(months)
    ax.set_xlabel("month")
    ax.set_ylabel("0=ranging 1=trend")
    ax.set_title(f"XAUUSD {year} seasonality (calendar-month)")
    ax.legend(ncol=5)
    fig.tight_layout()
    out = OUT_DIR / f"xauusd_{year}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    feat = pl.read_parquet(FEAT).sort(["year", "month"])
    years = sorted(set(feat["year"].to_list()))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for y in years:
        out = draw_year(y, feat)
        print(f"{y}: months={feat.filter(pl.col('year') == y).height} -> {out}")
    print(f"done: {len(years)} charts in {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
