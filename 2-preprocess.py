"""Fill missing XAUUSD daily candles with flat candles in a Parquet dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

REQUIRED_COLUMNS = {
    "time",
    "open",
    "high",
    "low",
    "close",
    "tick_volume",
    "spread",
    "real_volume",
}
OUTPUT_COLUMNS = ["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
OHLC_COLUMNS = ["open", "high", "low", "close"]
DEFAULT_INPUT = Path("XAUUSD_D1.parquet")
DEFAULT_OUTPUT = Path("XAUUSD_D1_clean.parquet")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill missing weekday D1 candles using the previous close."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="source Parquet file")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="cleaned Parquet file")
    return parser.parse_args()


def fill_missing_candles(data: pl.DataFrame) -> tuple[pl.DataFrame, int]:
    """Return weekday-complete data, filling missing/invalid candles from prior close."""
    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing_columns))}")
    if data.is_empty():
        raise ValueError("Input data contains no candles.")

    data = data.sort("time").with_columns(pl.col("time").cast(pl.Datetime).dt.date().alias("_date"))
    if data.select(pl.col("_date").is_duplicated().any()).item():
        raise ValueError("Input has more than one candle for at least one date.")

    dates = pl.date_range(data["_date"].min(), data["_date"].max(), interval="1d", eager=True)
    dates = dates.filter(dates.dt.weekday() <= 5)
    calendar = pl.DataFrame({"_date": dates})
    merged = calendar.join(data, on="_date", how="left").sort("_date")

    # Any absent candle, or a candle with an absent OHLC value, is normalized to
    # the latest valid close. Volume and spread are zero for synthesized candles.
    is_empty = pl.any_horizontal([pl.col(column).is_null() for column in OHLC_COLUMNS])
    previous_close = pl.when(~is_empty).then(pl.col("close")).otherwise(None).forward_fill()
    if merged.select(previous_close.is_null().any()).item():
        raise ValueError("The first weekday candle is empty; no previous close is available.")

    cleaned = (
        merged.with_columns(
            is_empty.alias("_is_empty"),
            previous_close.alias("_previous_close"),
        )
        .with_columns(
            pl.when("_is_empty")
            .then(pl.col("_previous_close"))
            .otherwise(pl.col(column))
            .alias(column)
            for column in OHLC_COLUMNS
        )
        .with_columns(
            pl.when("_is_empty").then(0).otherwise(pl.col(column)).alias(column)
            for column in ("tick_volume", "spread", "real_volume")
        )
        .with_columns(pl.col("_date").cast(pl.Datetime).alias("time"))
    )
    filled_count = cleaned.select(pl.col("_is_empty").sum()).item()
    return cleaned.select(OUTPUT_COLUMNS), filled_count


def main() -> int:
    args = parse_arguments()
    try:
        cleaned, filled_count = fill_missing_candles(pl.read_parquet(args.input))
    except (OSError, ValueError, pl.exceptions.PolarsError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cleaned.write_parquet(args.output, compression="zstd")
    print(
        f"Saved {cleaned.height:,} candles to {args.output}; filled {filled_count:,} empty candles."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
