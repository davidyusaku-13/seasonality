"""Check earliest available XAUUSD D1 bar without downloading full history.

Strategy: probe single bars via copy_rates_from_pos with exponential +
binary search. Each probe downloads 1 bar only, so we never pull full OHLC.
"""

import sys
from datetime import datetime, timedelta, timezone

import holidays
import MetaTrader5 as mt5

TARGET = "XAUUSD"
TIMEFRAME = mt5.TIMEFRAME_D1


def fail(msg: str) -> None:
    print(f"ERROR: {msg}")
    err = mt5.last_error()
    print(f"last_error={err}")
    mt5.shutdown()
    sys.exit(1)


def get_one(symbol: str, pos: int):
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, pos, 1)
    if rates is None or len(rates) == 0:
        return None
    return rates[0]


def find_symbol() -> str:
    # exact match first
    info = mt5.symbol_info(TARGET)
    if info is not None:
        return TARGET
    # fallback: search variants like XAUUSD., XAUUSDm, XAUUSD+, GOLD, etc.
    group_patterns = ["*XAUUSD*", "*XAU*", "*GOLD*"]
    candidates = []
    for pat in group_patterns:
        found = mt5.symbol_get_group(pat)
        if found:
            candidates.extend(s.name for s in found)
    # dedupe, prefer names starting with XAUUSD
    candidates = sorted(set(candidates))
    print(f"Exact '{TARGET}' not found. Candidates: {candidates}")
    for name in candidates:
        if name.startswith("XAUUSD"):
            return name
    if candidates:
        return candidates[0]
    fail(f"no symbol matching {TARGET}. Are you logged in / connected?")
    raise AssertionError


def check_no_missing_days(symbol: str, total_bars: int) -> bool:
    """Fetch full D1 series (~5k bars, tiny) and verify no missing weekdays.

    Returns True if clean, False otherwise. Prints full report.
    A 'missing day' = Mon-Fri date with no D1 bar, excluding market
    holidays from the `holidays` lib (NYSE calendar: Christmas, New Year,
    Good Friday + observed days). Weekends are expected to have no bars;
    any Sat/Sun bar is reported separately (this broker emits Sunday bars).
    """
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, total_bars)
    if rates is None or len(rates) == 0:
        print(f"ERROR: copy_rates_from_pos({symbol}, D1, 0, {total_bars}) returned nothing")
        print(f"last_error={mt5.last_error()}")
        return False
    print(f"fetched {len(rates)} D1 bars for gap check (requested {total_bars})")
    if len(rates) != total_bars:
        print(f"WARNING: fetched {len(rates)} != probed total {total_bars} (history changed?)")

    times = [datetime.fromtimestamp(int(r["time"]), tz=timezone.utc) for r in rates]
    # MT5 returns newest-first for copy_rates_from_pos; sort oldest-first
    order = sorted(range(len(times)), key=lambda i: times[i])
    times = [times[i] for i in order]
    rates = [rates[i] for i in order]

    ok = True

    # 1. duplicates / ordering / OHLC sanity
    seen = set()
    dups = []
    bad_ohlc = []
    for r, t in zip(rates, times):
        d = t.date()
        if d in seen:
            dups.append(d)
        seen.add(d)
        if not (r["high"] >= r["low"] and r["high"] >= r["open"] and r["high"] >= r["close"]
                and r["low"] <= r["open"] and r["low"] <= r["close"]):
            bad_ohlc.append((d, r["open"], r["high"], r["low"], r["close"]))
        if r["open"] <= 0 or r["high"] <= 0 or r["low"] <= 0 or r["close"] <= 0:
            bad_ohlc.append((d, r["open"], r["high"], r["low"], r["close"]))
    if dups:
        ok = False
        print(f"FAIL: {len(dups)} duplicate dates, e.g. {dups[:10]}")
    else:
        print("PASS: no duplicate dates")
    if bad_ohlc:
        ok = False
        print(f"FAIL: {len(bad_ohlc)} bad OHLC rows, e.g. {bad_ohlc[:5]}")
    else:
        print("PASS: OHLC sanity (H>=O,H,L,C>=L, all>0)")

    # 2. weekend bars (unexpected)
    weekend_bars = [t.date() for t in times if t.weekday() >= 5]
    if weekend_bars:
        print(f"NOTE: {len(weekend_bars)} weekend bars present, e.g. {weekend_bars[:10]}")
    else:
        print("PASS: no weekend (Sat/Sun) bars")

    # 3. missing weekdays across full calendar range, ignoring holidays
    start, end = times[0].date(), times[-1].date()
    missing = []
    day = start
    while day <= end:
        if day.weekday() < 5 and day not in seen:
            missing.append(day)
        day += timedelta(days=1)

    market_holidays = holidays.financial_holidays("NYSE", years=range(start.year, end.year + 1))
    holiday_missing = [(d, market_holidays.get(d)) for d in missing if d in market_holidays]
    unexpected = [d for d in missing if d not in market_holidays]

    if holiday_missing:
        print(f"NOTE: {len(holiday_missing)} missing weekday(s) are NYSE market holidays (ignored):")
        for d, name in holiday_missing:
            print(f"  holiday: {d} ({d.strftime('%a')}) - {name}")
    if unexpected:
        ok = False
        print(f"FAIL: {len(unexpected)} UNEXPECTED missing weekday(s) (not holidays):")
        for d in unexpected:
            print(f"  missing: {d} ({d.strftime('%a')})")
    elif not missing:
        print(f"PASS: no missing weekdays between {start} and {end} ({len(seen)} trading days)")
    else:
        print(f"PASS: no unexpected gaps ({len(missing)} missing all explained as holidays)")

    # 4. consecutive-gap breakdown (helps distinguish holidays vs data holes)
    gaps = []
    for a, b in zip(times, times[1:]):
        delta = (b.date() - a.date()).days
        if delta <= 1:
            continue
        # count intervening weekdays
        inter_weekdays = sum(
            1 for i in range(1, delta)
            if (a.date() + timedelta(days=i)).weekday() < 5
        )
        if inter_weekdays > 0:
            gaps.append((a.date(), b.date(), delta, inter_weekdays))
    if gaps:
        print(f"NOTE: {len(gaps)} gaps with missing weekdays (likely holidays/halts), largest:")
        for g in sorted(gaps, key=lambda x: -x[3])[:10]:
            print(f"  {g[0]} -> {g[1]}: {g[2]}d apart, {g[3]} weekday(s) missing")
    else:
        print("PASS: consecutive bars have no weekday gaps beyond weekends")

    print("GAP-CHECK:", "PASS - no missing days" if ok else "FAIL - missing/duplicates found")
    return ok


def main() -> None:
    if not mt5.initialize():
        fail("mt5.initialize() failed. Is MT5 terminal running and logged in?")

    print(f"terminal={mt5.terminal_info()}")
    print(f"account={mt5.account_info()}")

    symbol = find_symbol()
    print(f"using symbol={symbol}")

    info = mt5.symbol_info(symbol)
    print(f"visible={info.visible} trade_mode={info.trade_mode}")

    if not info.visible:
        print(f"enabling {symbol} in MarketWatch...")
        if not mt5.symbol_select(symbol, True):
            fail(f"symbol_select({symbol}) failed")

    # latest bar (pos 0)
    latest = get_one(symbol, 0)
    if latest is None:
        fail(f"no D1 data at pos 0 for {symbol}")
    latest_time = datetime.fromtimestamp(int(latest["time"]), tz=timezone.utc)
    print(f"latest D1 bar: time_utc={latest_time} O={latest['open']} H={latest['high']} L={latest['low']} C={latest['close']}")

    # exponential search for upper bound of history
    lo = 0  # known-good position
    hi = 1
    while True:
        bar = get_one(symbol, hi)
        if bar is None:
            break
        lo = hi
        print(f"pos {hi} ok -> {datetime.fromtimestamp(int(bar['time']), tz=timezone.utc)}")
        hi *= 2
        if hi > 5_000_000:  # sanity cap (~13k years of D1)
            break
        # safety: D1 history realistically < 100k bars
        if hi > 100_000:
            # keep going but slower? just break to binary search
            # actually continue until fail to be exact
            pass

    print(f"history bound: last_ok={lo}, first_fail={hi}")

    # binary search max valid position in (lo, hi)
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        bar = get_one(symbol, mid)
        if bar is None:
            hi = mid
        else:
            lo = mid

    earliest = get_one(symbol, lo)
    earliest_time = datetime.fromtimestamp(int(earliest["time"]), tz=timezone.utc)
    print(f"total D1 bars available: ~{lo + 1}")
    print(f"earliest D1 bar: pos={lo} time_utc={earliest_time} O={earliest['open']} H={earliest['high']} L={earliest['low']} C={earliest['close']} tick_volume={earliest['tick_volume']}")
    # also show server-local time for clarity
    print(f"earliest date (UTC): {earliest_time.date()}")

    total = lo + 1
    clean = check_no_missing_days(symbol, total)

    mt5.shutdown()
    if not clean:
        sys.exit(2)


if __name__ == "__main__":
    main()
