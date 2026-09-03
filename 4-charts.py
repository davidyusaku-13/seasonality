"""Yearly XAUUSD drawings: price + monthly seasonality per year.

Same two-panel layout as tmp.py (top: daily close with month separators,
bottom: monthly S_m 0-1 + components), one PNG per year.

Outputs: charts/yearly/xauusd_YYYY.png for 2007..2026.
Features come from data/monthly_features.parquet (no recompute);
prices from data/xauusd_d1.parquet (real bars only).
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import polars as pl

matplotlib.rcParams["axes.grid"] = False

SRC = Path("data/xauusd_d1.parquet")
FEAT = Path("data/monthly_features.parquet")
OUT_DIR = Path("charts/yearly")


def draw_year(year: int, real: pl.DataFrame, feat: pl.DataFrame) -> Path:
    yd = real.filter(pl.col("date").dt.year() == year).sort("time")
    yf = feat.filter(pl.col("year") == year).sort("month")
    if yd.height == 0:
        raise SystemExit(f"no rows for {year}")
    dates = yd["date"].to_list()
    closes = yd["close"].to_list()
    months = yf["month"].to_list()

    fig, (ax_p, ax_s) = plt.subplots(2, 1, figsize=(14, 8),
                                     gridspec_kw={"height_ratios": [3, 1]})
    ax_p.plot(dates, closes, linewidth=1.2, label="close")
    seen: dict = {}
    for d in dates:
        seen.setdefault(d.month, d)
    for mm in sorted(seen):
        ax_p.axvline(seen[mm], color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax_p.axvline(dates[-1], color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax_p.set_title(f"XAUUSD D1 {year} (vertical = month start)")
    ax_p.set_ylabel("close (USD)")
    ax_p.legend()

    sm = yf["s"].to_list()
    ax_s.bar(months, sm, width=0.7, label="S_m")
    ax_s.plot(months, yf["t_range"].to_list(), marker="o", linewidth=1, label="T_range")
    ax_s.plot(months, yf["t_direction"].to_list(), marker="o", linewidth=1, label="T_dir")
    ax_s.plot(months, yf["t_mono"].to_list(), marker="o", linewidth=1, label="T_mono")
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
    out = OUT_DIR / f"xauusd_{year}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    real = pl.read_parquet(SRC).filter(~pl.col("is_filled")).sort("time")
    feat = pl.read_parquet(FEAT).sort(["year", "month"])
    years = sorted(set(real["date"].dt.year().to_list()))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for y in years:
        out = draw_year(y, real, feat)
        n = real.filter(pl.col("date").dt.year() == y).height
        print(f"{y}: rows={n} -> {out}")
    print(f"done: {len(years)} charts in {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
