"""XAUUSD 2025: close line + seasonality together (mirrors 4-charts.py)."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import polars as pl

matplotlib.rcParams["axes.grid"] = False

SRC = Path("data/xauusd_d1.parquet")
FEAT = Path("data/monthly_features.parquet")
OUT = Path("charts/yearly/xauusd_2025.png")
YEAR = 2025
SHOW_LINES = False  # set True to re-show q_m / T lines (hidden, not removed)


def main() -> None:
    yd = (
        pl.read_parquet(SRC)
        .filter((~pl.col("is_filled")) & (pl.col("date").dt.year() == YEAR))
        .sort("time")
    )
    yf = pl.read_parquet(FEAT).filter(pl.col("year") == YEAR).sort("month")
    if yd.height == 0:
        raise SystemExit(f"no rows for {YEAR}")
    dates = yd["date"].to_list()
    closes = yd["close"].to_list()
    months = yf["month"].to_list()

    fig, (ax_p, ax_s) = plt.subplots(
        2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]}
    )
    ax_p.plot(dates, closes, linewidth=1.2, label="close")
    ax_p.legend()
    seen: dict = {}
    for d in dates:
        seen.setdefault(d.month, d)
    for mm in sorted(seen):
        ax_p.axvline(seen[mm], color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax_p.axvline(dates[-1], color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax_p.set_xlim(dates[0], dates[-1])
    ax_p.set_title(f"XAUUSD D1 {YEAR} (vertical = month separator)")
    ax_p.set_ylabel("close (USD)")

    bars = ax_s.bar(months, yf["s"].to_list(), width=0.7, alpha=0.85, label="S_m")
    for rect, v in zip(bars, yf["s"].to_list()):
        h = rect.get_height()
        x = rect.get_x() + rect.get_width() / 2
        if h > 0.12:
            ax_s.text(
                x,
                h / 2,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
            )
        else:
            ax_s.text(
                x,
                h + 0.015,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="black",
            )
    ax_s.plot(
        months,
        yf["q_trend"].to_list(),
        marker="o",
        linewidth=2,
        color="black",
        label="q_m" if SHOW_LINES else "_q_m",
        visible=SHOW_LINES,
    )
    ax_s.plot(
        months,
        yf["t_range"].to_list(),
        marker=".",
        linewidth=1,
        alpha=0.7,
        label="T_range" if SHOW_LINES else "_T_range",
        visible=SHOW_LINES,
    )
    ax_s.plot(
        months,
        yf["t_direction"].to_list(),
        marker=".",
        linewidth=1,
        alpha=0.7,
        label="T_dir" if SHOW_LINES else "_T_dir",
        visible=SHOW_LINES,
    )
    ax_s.plot(
        months,
        yf["t_mono"].to_list(),
        marker=".",
        linewidth=1,
        alpha=0.7,
        label="T_mono" if SHOW_LINES else "_T_mono",
        visible=SHOW_LINES,
    )
    for mm in months:
        ax_s.axvline(mm - 0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax_s.axvline(
        months[-1] + 0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7
    )
    ax_s.set_ylim(0, 1)
    ax_s.set_xlim(months[0] - 0.5, 12.5)
    ax_s.set_xticks(months)
    ax_s.set_xlabel("month")
    ax_s.set_ylabel("0=ranging 1=trend")
    ax_s.set_title("Seasonality")
    ax_s.legend(ncol=5)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"saved={OUT.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()
