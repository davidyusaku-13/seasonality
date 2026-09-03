"""Download full XAUUSD D1 history to parquet (polars).

- Finds total available bars via single-bar probe (no guesswork).
- Downloads all D1 bars at once (~5k rows, tiny).
- Builds full Mon-Fri calendar 2007-06-22..latest, fills missing weekdays
  (holidays + 2022-12-01 gap) with flat bars: O=H=L=C=prev close, volume 0.
- Keeps real weekend (Sunday) bars as-is; only fills missing weekdays.
- Output: data/xauusd_d1.parquet with is_filled / is_holiday flags.
"""

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import holidays
import MetaTrader5 as mt5
import polars as pl

TARGET = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_D1
OUT = Path("data/xauusd_d1.parquet")


def fail(msg: str) -> None:
    print(f"ERROR: {msg} last_error={mt5.last_error()}")
    mt5.shutdown()
    sys.exit(1)


def get_one(symbol: str, pos: int):
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, pos, 1)
    if rates is None or len(rates) == 0:
        return None
    return rates[0]


def find_total(symbol: str) -> int:
    lo, hi = 0, 1
    while get_one(symbol, hi) is not None:
        lo = hi
        hi *= 2
        if hi > 1_000_000:
            break
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if get_one(symbol, mid) is None:
            hi = mid
        else:
            lo = mid
    return lo + 1


def main() -> None:
    if not mt5.initialize():
        fail("mt5.initialize() failed")
    info = mt5.symbol_info(TARGET)
    if info is None:
        fail(f"symbol {TARGET} not found")
    if not info.visible and not mt5.symbol_select(TARGET, True):
        fail(f"symbol_select({TARGET}) failed")

    total = find_total(TARGET)
    print(f"total D1 bars on server: {total}")

    rates = mt5.copy_rates_from_pos(TARGET, TIMEFRAME, 0, total)
    if rates is None or len(rates) == 0:
        fail("copy_rates_from_pos returned nothing")
    print(f"downloaded {len(rates)} bars")
    mt5.shutdown()

    df = (
        pl.DataFrame(
            {
                "time": [datetime.fromtimestamp(int(r["time"]), tz=UTC) for r in rates],
                "open": [float(r["open"]) for r in rates],
                "high": [float(r["high"]) for r in rates],
                "low": [float(r["low"]) for r in rates],
                "close": [float(r["close"]) for r in rates],
                "tick_volume": [int(r["tick_volume"]) for r in rates],
                "spread": [int(r["spread"]) for r in rates],
                "real_volume": [int(r["real_volume"]) for r in rates],
            }
        )
        .with_columns(pl.col("time").dt.date().alias("date"))
        .sort("time")
    )

    # NYSE holidays for flagging (Christmas, New Year, Good Friday, observed)
    start_d, end_d = df["date"].min(), df["date"].max()
    nyse = holidays.financial_holidays(
        "NYSE", years=range(start_d.year, end_d.year + 1)
    )

    df = df.with_columns(
        pl.lit(False).alias("is_filled"),
        pl.col("date").is_in(list(nyse.keys())).alias("is_holiday"),
    )

    have = set(df["date"].to_list())
    close_by_date = dict(zip(df["date"].to_list(), df["close"].to_list()))

    missing: list = []
    day = start_d
    while day <= end_d:
        if day.weekday() < 5 and day not in have:
            missing.append(day)
        day += timedelta(days=1)
    print(f"missing weekdays: {len(missing)}")

    if missing:
        # prev close lookup: latest close strictly before the gap (chains for consecutive gaps)
        sorted_dates = sorted(have)
        filled_rows = []
        for d in sorted(missing):
            prev = max(x for x in sorted_dates if x < d)
            c = close_by_date[prev]
            filled_rows.append((d, c))
            close_by_date[d] = c  # allow chaining for consecutive missing
            sorted_dates.append(d)

        fill_df = pl.DataFrame(
            {
                "time": [
                    datetime(d.year, d.month, d.day, tzinfo=UTC) for d, _ in filled_rows
                ],
                "open": [c for _, c in filled_rows],
                "high": [c for _, c in filled_rows],
                "low": [c for _, c in filled_rows],
                "close": [c for _, c in filled_rows],
                "tick_volume": [0] * len(filled_rows),
                "spread": [0] * len(filled_rows),
                "real_volume": [0] * len(filled_rows),
                "date": [d for d, _ in filled_rows],
                "is_filled": [True] * len(filled_rows),
                "is_holiday": [d in nyse for d, _ in filled_rows],
            },
            schema=df.schema,
        )
        df = pl.concat([df, fill_df]).sort("time")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT)
    n_filled = int(df["is_filled"].sum())
    print(f"wrote {OUT}: rows={df.height} filled={n_filled}")
    print(f"range {df['date'].min()} .. {df['date'].max()}")
    for filled, holiday, n in sorted(
        df.group_by(["is_filled", "is_holiday"]).len().rows()
    ):
        print(f"is_filled={filled} is_holiday={holiday} n={n}")


if __name__ == "__main__":
    main()
