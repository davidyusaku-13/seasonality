"""Create one clean XAUUSD D1 close-price line chart for each year."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl


DEFAULT_INPUT = Path("XAUUSD_D1_clean.parquet")
DEFAULT_OUTPUT_DIR = Path("output/visualizations/xauusd_d1")
REQUIRED_COLUMNS = {"time", "close"}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one XAUUSD D1 close-price chart per year."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="source Parquet file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for yearly PNG files",
    )
    parser.add_argument("--dpi", type=int, default=160, help="PNG resolution (default: 160)")
    return parser.parse_args()


def load_prices(input_path: Path) -> pl.DataFrame:
    """Load, validate, and sort the daily close-price history."""
    data = pl.read_parquet(input_path)
    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing_columns))}")
    if data.is_empty():
        raise ValueError("Input data contains no candles.")
    return (
        data.select("time", "close")
        .drop_nulls()
        .sort("time")
        .with_columns(pl.col("time").dt.year().alias("year"), pl.col("time").dt.month().alias("month"))
    )


def save_year_chart(year_data: pl.DataFrame, output_path: Path, dpi: int) -> None:
    """Save a yearly close line chart with month boundaries as separators."""
    year = year_data["year"][0]
    dates = year_data["time"].to_list()
    closes = year_data["close"].to_list()
    month_starts = (
        year_data.group_by("month", maintain_order=True)
        .agg(pl.col("time").first())
        .get_column("time")
        .to_list()
    )

    figure, axis = plt.subplots(figsize=(16, 7), constrained_layout=True)
    axis.plot(dates, closes, color="#b8860b", linewidth=1.25)
    for separator in month_starts[1:]:
        axis.axvline(separator, color="#8a8a8a", linewidth=0.7, linestyle="--", alpha=0.7)

    axis.set_title(f"XAUUSD D1 Close — {year}", fontweight="bold")
    axis.set_ylabel("Price")
    axis.set_xlabel("Month")
    axis.set_xticks(month_starts)
    axis.set_xticklabels([date.strftime("%b") for date in month_starts])
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.6)
    axis.margins(x=0.01)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def create_yearly_charts(data: pl.DataFrame, output_dir: Path, dpi: int) -> list[Path]:
    """Write one named PNG per calendar year into a single output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for partitions in data.partition_by("year", as_dict=False, maintain_order=True):
        year = partitions["year"][0]
        output_path = output_dir / f"XAUUSD_D1_{year}.png"
        save_year_chart(partitions, output_path, dpi)
        outputs.append(output_path)
    return outputs


def main() -> int:
    args = parse_arguments()
    if args.dpi <= 0:
        print("Error: --dpi must be positive.", file=sys.stderr)
        return 1

    try:
        data = load_prices(args.input)
        outputs = create_yearly_charts(data, args.output_dir, args.dpi)
    except (OSError, ValueError, pl.exceptions.PolarsError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Saved {len(outputs)} yearly charts in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
