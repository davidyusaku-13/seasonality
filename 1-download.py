"""Download all available XAUUSD daily OHLC data from MetaTrader 5.

MetaTrader 5 must be installed, running, and logged in to a broker account before
this script is run. The earliest returned row depends on the broker's history and
the terminal's "Max. bars in chart" setting.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import MetaTrader5 as mt5
import polars as pl


DEFAULT_SYMBOL = "XAUUSD"
DEFAULT_OUTPUT = Path("XAUUSD_D1.parquet")
EARLIEST_DATE = datetime(1970, 1, 1, tzinfo=UTC)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download all available XAUUSD D1 OHLC data from MetaTrader 5."
    )
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help="MT5 symbol name")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="destination Parquet file (default: XAUUSD_D1.parquet)",
    )
    return parser.parse_args()


def download_ohlc(symbol: str) -> pl.DataFrame:
    """Return all daily OHLC bars currently available in the connected terminal."""
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialization failed: {mt5.last_error()}")

    try:
        info = mt5.symbol_info(symbol)
        if info is None:
            raise ValueError(
                f"Symbol {symbol!r} was not found. Check the broker's Market Watch name."
            )
        if not info.visible and not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"Could not select {symbol!r}: {mt5.last_error()}")

        # A very early UTC start requests the full history exposed by the broker.
        # MT5 returns only the portion available locally/in the broker history.
        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_D1, EARLIEST_DATE, datetime.now(UTC))
        if rates is None:
            raise RuntimeError(f"Failed to download rates: {mt5.last_error()}")
        if len(rates) == 0:
            raise RuntimeError(f"No D1 history is available for {symbol!r}.")

        # Construct columns directly from MT5's NumPy structured array, then keep
        # the compact, typed representation through the Parquet write.
        data = pl.DataFrame({field: rates[field] for field in rates.dtype.names})
        return (
            # MT5 timestamps are Unix timestamps (UTC). Keep them as timezone-naive
            # UTC datetimes for portable Parquet reads, including on Windows.
            data.with_columns(pl.from_epoch("time", time_unit="s"))
            .select(
                "time",
                "open",
                "high",
                "low",
                "close",
                "tick_volume",
                "spread",
                "real_volume",
            )
        )
    finally:
        mt5.shutdown()


def main() -> int:
    args = parse_arguments()
    try:
        data = download_ohlc(args.symbol)
    except (RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.write_parquet(args.output, compression="zstd")
    print(
        f"Saved {data.height:,} {args.symbol} D1 bars "
        f"({data['time'][0].date()} to {data['time'][-1].date()}) to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
