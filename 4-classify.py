"""Classify each month of one year of XAUUSD D1 data as trending or ranging."""

from __future__ import annotations

import argparse
import calendar
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


DEFAULT_INPUT = Path("XAUUSD_D1_clean.parquet")
DEFAULT_OUTPUT_DIR = Path("output/classifications/xauusd_d1")
R_SQUARED_THRESHOLD = 0.35
EFFICIENCY_RATIO_THRESHOLD = 0.40
TRENDING_COLOR = "#2e8b57"
RANGING_COLOR = "#d97706"
INSUFFICIENT_DATA_COLOR = "#9ca3af"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify each month of XAUUSD D1 data as trending or ranging."
    )
    parser.add_argument("--year", type=int, help="analyze only this year (default: all available years)")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="source Parquet file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for classification plots",
    )
    parser.add_argument("--dpi", type=int, default=160, help="PNG resolution (default: 160)")
    return parser.parse_args()


def classify_month(data: pl.DataFrame) -> dict[str, float | str | int]:
    """Classify a month using regression linearity and price-move efficiency."""
    closes = data["close"].to_numpy().astype(float)
    if len(closes) < 10:
        return {
            "candles": len(closes),
            "r_squared": float("nan"),
            "efficiency_ratio": float("nan"),
            "verdict": "INSUFFICIENT DATA",
        }
    if np.any(closes <= 0):
        raise ValueError("Close prices must be positive.")

    positions = np.arange(len(closes), dtype=float)
    # Log prices make the linear fit comparable across different price levels.
    log_closes = np.log(closes)
    slope, intercept = np.polyfit(positions, log_closes, 1)
    predicted = slope * positions + intercept
    residual_sum = np.sum((log_closes - predicted) ** 2)
    total_sum = np.sum((log_closes - log_closes.mean()) ** 2)
    r_squared = 1.0 if total_sum == 0 else max(0.0, 1.0 - residual_sum / total_sum)

    total_path = np.abs(np.diff(closes)).sum()
    efficiency_ratio = 0.0 if total_path == 0 else abs(closes[-1] - closes[0]) / total_path
    verdict = (
        "TRENDING"
        if r_squared >= R_SQUARED_THRESHOLD and efficiency_ratio >= EFFICIENCY_RATIO_THRESHOLD
        else "RANGING"
    )

    return {
        "candles": len(closes),
        "r_squared": r_squared,
        "efficiency_ratio": efficiency_ratio,
        "verdict": verdict,
    }


def save_classification_plot(results: list[dict[str, float | str | int]], output_path: Path, year: int, dpi: int) -> None:
    """Save R-squared and efficiency bars, colored by the monthly regime."""
    months = [calendar.month_abbr[int(result["month"])] for result in results]
    r_squared = [float(result["r_squared"]) for result in results]
    efficiency = [float(result["efficiency_ratio"]) for result in results]
    colors = [
        TRENDING_COLOR
        if result["verdict"] == "TRENDING"
        else RANGING_COLOR
        if result["verdict"] == "RANGING"
        else INSUFFICIENT_DATA_COLOR
        for result in results
    ]
    positions = np.arange(len(months))

    figure, (r_squared_axis, efficiency_axis) = plt.subplots(
        2, 1, figsize=(15, 8), sharex=True, constrained_layout=True
    )
    for axis, values, threshold, title in (
        (r_squared_axis, r_squared, R_SQUARED_THRESHOLD, "R-squared (linearity)"),
        (efficiency_axis, efficiency, EFFICIENCY_RATIO_THRESHOLD, "Efficiency ratio (directional movement)"),
    ):
        axis.bar(positions, values, color=colors, width=0.72)
        axis.axhline(threshold, color="#4b5563", linewidth=1, linestyle="--", label=f"Threshold {threshold:.2f}")
        axis.set_ylim(0, 1)
        axis.set_ylabel(title)
        axis.grid(axis="y", color="#e5e7eb", linewidth=0.7)
        axis.legend(loc="upper right")

    efficiency_axis.set_xticks(positions)
    efficiency_axis.set_xticklabels(months)
    figure.suptitle(
        f"XAUUSD D1 Monthly Classification - {year}\n"
        "Green: TRENDING   Orange: RANGING   Gray: INSUFFICIENT DATA",
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = parse_arguments()
    if args.dpi <= 0:
        print("Error: --dpi must be positive.", file=sys.stderr)
        return 1
    try:
        data = (
            pl.read_parquet(args.input)
            .select("time", "close")
            .drop_nulls()
            .sort("time")
            .with_columns(
                pl.col("time").dt.year().alias("year"),
                pl.col("time").dt.month().alias("month"),
            )
        )
        if args.year is not None:
            data = data.filter(pl.col("year") == args.year)
        if data.is_empty():
            requested_period = args.year if args.year is not None else "any year"
            raise ValueError(f"No candles found for {requested_period}.")
    except (OSError, ValueError, pl.exceptions.PolarsError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    yearly_data = data.partition_by("year", as_dict=False, maintain_order=True)
    for year_data in yearly_data:
        year = int(year_data["year"][0])
        print(f"XAUUSD D1 monthly regime - {year}")
        print(
            f"R^2 threshold: >= {R_SQUARED_THRESHOLD:.3f}; "
            f"Efficiency threshold: >= {EFFICIENCY_RATIO_THRESHOLD:.3f}"
        )
        print("Month      Candles  R^2    Efficiency  Verdict")
        print("---------  -------  -----  ----------  --------")
        results: list[dict[str, float | str | int]] = []
        for month_data in year_data.partition_by("month", as_dict=False, maintain_order=True):
            month = month_data["month"][0]
            result = classify_month(month_data)
            result["month"] = int(month)
            results.append(result)
            print(
                f"{calendar.month_name[month]:<9}  "
                f"{result['candles']:>7}  "
                f"{result['r_squared']:>5.3f}  "
                f"{result['efficiency_ratio']:>10.3f}  "
                f"{result['verdict']}"
            )
        output_path = args.output_dir / f"XAUUSD_D1_{year}_classification.png"
        save_classification_plot(results, output_path, year, args.dpi)
        print(f"Saved classification plot to {output_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
