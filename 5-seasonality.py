"""Describe XAUUSD D1 calendar-month seasonality from historical regime data."""

from __future__ import annotations

import argparse
import calendar
import importlib.util
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

DEFAULT_INPUT = Path("XAUUSD_D1_clean.parquet")


def load_classifier():
    """Load the production rules so historical labels remain consistent."""
    classifier_path = Path(__file__).with_name("4-classify.py")
    spec = importlib.util.spec_from_file_location("monthly_classifier", classifier_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load classifier at {classifier_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Describe historical XAUUSD D1 regimes for each calendar month."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="source Parquet file")
    return parser.parse_args()


def wilson_interval(successes: int, total: int, z_score: float = 1.96) -> tuple[float, float]:
    """Return a 95% binomial confidence interval suitable for small samples."""
    proportion = successes / total
    denominator = 1 + z_score**2 / total
    center = (proportion + z_score**2 / (2 * total)) / denominator
    margin = (
        z_score
        * math.sqrt(proportion * (1 - proportion) / total + z_score**2 / (4 * total**2))
        / denominator
    )
    return center - margin, center + margin


def collect_monthly_statistics(
    data: pl.DataFrame, classifier
) -> dict[int, list[dict[str, float | str]]]:
    """Classify complete months, keeping only metrics known for that month."""
    statistics: dict[int, list[dict[str, float | str]]] = defaultdict(list)
    for candles in data.partition_by(["year", "month"], as_dict=False, maintain_order=True):
        result = classifier.classify_month(candles)
        if result["verdict"] == "INSUFFICIENT DATA":
            continue
        closes = candles["close"].to_numpy().astype(float)
        month = int(candles["month"][0])
        statistics[month].append(
            {
                "verdict": str(result["verdict"]),
                "r_squared": float(result["r_squared"]),
                "efficiency_ratio": float(result["efficiency_ratio"]),
                "return_pct": (closes[-1] / closes[0] - 1.0) * 100,
            }
        )
    return statistics


def main() -> int:
    args = parse_arguments()
    try:
        classifier = load_classifier()
        data = (
            pl.read_parquet(args.input)
            .select("time", "close")
            .drop_nulls()
            .sort("time")
            .with_columns(
                pl.col("time").dt.year().alias("year"), pl.col("time").dt.month().alias("month")
            )
        )
        statistics = collect_monthly_statistics(data, classifier)
    except (OSError, RuntimeError, ValueError, pl.exceptions.PolarsError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print("XAUUSD D1 calendar-month seasonality (descriptive, not a forecast)")
    print("Month      Years  Trend rate (95% CI)  Median R^2  Median ER  Median return")
    print("---------  -----  -------------------  ----------  ---------  -------------")
    for month in range(1, 13):
        values = statistics[month]
        total = len(values)
        trending = sum(value["verdict"] == "TRENDING" for value in values)
        lower, upper = wilson_interval(trending, total)
        median_r_squared = np.median([float(value["r_squared"]) for value in values])
        median_er = np.median([float(value["efficiency_ratio"]) for value in values])
        median_return = np.median([float(value["return_pct"]) for value in values])
        print(
            f"{calendar.month_name[month]:<9}  "
            f"{total:>5}  "
            f"{trending / total:>6.1%} ({lower:.1%}-{upper:.1%})  "
            f"{median_r_squared:>10.3f}  "
            f"{median_er:>9.3f}  "
            f"{median_return:>+12.2f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
