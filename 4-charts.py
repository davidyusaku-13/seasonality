"""Yearly drawings: candlesticks + calendar-month seasonality together.

Top: daily candlesticks with month separators. Bottom: S_m bars, q_m line,
T components (0-1). One PNG per year in charts/yearly/xauusd_YYYY.png.
Prices from data/xauusd_d1.parquet (real bars); features from
data/monthly_features.parquet (no recompute).
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import polars as pl

matplotlib.rcParams["axes.grid"] = False

SRC = Path("data/xauusd_d1.parquet")
FEAT = Path("data/monthly_features.parquet")
OUT_DIR = Path("charts/yearly")
SHOW_LINES = False  # set True to re-show q_m / T lines (hidden, not removed)


def draw_year(year: int, real: pl.DataFrame, feat: pl.DataFrame) -> Path:
    yd = real.filter(pl.col("date").dt.year() == year).sort("time")
    yf = feat.filter(pl.col("year") == year).sort("month")
    if yd.height == 0:
        raise SystemExit(f"no rows for {year}")
    dates = yd["date"].to_list()
    opens = yd["open"].to_list()
    highs = yd["high"].to_list()
    lows = yd["low"].to_list()
    closes = yd["close"].to_list()
    months = yf["month"].to_list()

    fig, (ax_p, ax_s) = plt.subplots(
        2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]}
    )
    up = [c >= o for c, o in zip(closes, opens)]
    dn = [not u for u in up]
    for flag, color, label in ((up, "green", "up"), (dn, "red", "down")):
        d = [x for x, f in zip(dates, flag) if f]
        o = [x for x, f in zip(opens, flag) if f]
        h = [x for x, f in zip(highs, flag) if f]
        lo = [x for x, f in zip(lows, flag) if f]
        c = [x for x, f in zip(closes, flag) if f]
        ax_p.vlines(d, lo, h, colors=color, linewidths=0.8)
        ax_p.bar(
            d,
            [ci - oi for ci, oi in zip(c, o)],
            bottom=o,
            width=0.6,
            color=color,
            label=label,
        )
    seen: dict = {}
    for d in dates:
        seen.setdefault(d.month, d)
    for mm in sorted(seen):
        ax_p.axvline(seen[mm], color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax_p.axvline(dates[-1], color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax_p.set_title(f"XAUUSD D1 {year} (vertical = month separator)")
    ax_p.set_ylabel("close (USD)")
    ax_p.legend()

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
    out = OUT_DIR / f"xauusd_{year}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    real = pl.read_parquet(SRC).filter(~pl.col("is_filled")).sort("time")
    feat = pl.read_parquet(FEAT).sort(["year", "month"])
    years = sorted(set(feat["year"].to_list()))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for y in years:
        out = draw_year(y, real, feat)
        print(f"{y} -> {out}")
    print(f"done: {len(years)} charts in {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
