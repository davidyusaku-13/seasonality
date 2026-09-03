"""XAUUSD 2025 seasonality as the main chart (calendar-month).

Mirrors 4-charts.py draw_year for a single year. Run all years with 4-charts.py.
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import polars as pl

matplotlib.rcParams["axes.grid"] = False

SRC = Path("data/monthly_features.parquet")
OUT = Path("charts/yearly/xauusd_2025.png")
YEAR = 2025


def main() -> None:
    yf = pl.read_parquet(SRC).filter(pl.col("year") == YEAR).sort("month")
    if yf.height == 0:
        raise SystemExit(f"no months for {YEAR} in {SRC}")
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
    ax.set_title(f"XAUUSD {YEAR} seasonality (calendar-month)")
    ax.legend(ncol=5)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"months={yf.height} saved={OUT.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()
