"""Validate monthly regime features against 10 controlled cases (FORMULA sec 21).

Builds synthetic 21-day D1 months with known structure, computes
[T_range, T_direction, T_mono, S_m] via 2-seasonality.month_features,
and checks the rankings agree with the intended trend-vs-sideways meaning.
Exit 0 if all checks pass, 1 otherwise.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

SPEC = Path(__file__).with_name("2-seasonality.py")
_spec = importlib.util.spec_from_file_location("seasonality2", SPEC)
seasonality2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(seasonality2)
month_features = seasonality2.month_features

N = 21
C0 = 3000.0


def ohlc_from_closes(closes: np.ndarray, wick: float = 0.0005):
    closes = np.asarray(closes, dtype=float)
    prev = np.concatenate([[C0], closes[:-1]])
    hi = np.maximum(closes, prev) * (1 + wick)
    lo = np.minimum(closes, prev) * (1 - wick)
    return hi, lo, closes


def t(days: int):
    return np.arange(1, days + 1)


CASES = {}


def case(name):
    def deco(fn):
        CASES[name] = fn
        return fn
    return deco


@case("smooth_bull")
def _():
    return ohlc_from_closes(C0 * 1.005 ** t(N))


@case("smooth_bear")
def _():
    return ohlc_from_closes(C0 * 0.995 ** t(N))


@case("quiet_sideways")
def _():
    rng = np.random.default_rng(11)
    c = C0 + 3 * np.sin(2 * np.pi * t(N) / 5) + rng.normal(0, 0.5, N).cumsum() * 0.1
    return ohlc_from_closes(c)


@case("volatile_sideways")
def _():
    alt = np.where(t(N) % 2 == 0, 1.02, 0.98)
    return ohlc_from_closes(C0 * alt, wick=0.005)


@case("shock")
def _():
    c = np.full(N, C0)
    c[10:] = C0 * 1.08
    return ohlc_from_closes(c)


@case("trend_reversal")
def _():
    up = C0 * 1.007 ** t(11)
    c = np.concatenate([up, up[-1] * 0.993 ** t(10)])
    return ohlc_from_closes(c)


@case("staircase")
def _():
    rets = np.tile([0.01, 0.01, 0.0], 7)
    return ohlc_from_closes(C0 * np.cumprod(1 + rets))


@case("trend_wicks")
def _():
    return ohlc_from_closes(C0 * 1.005 ** t(N), wick=0.015)


@case("random_walk")
def _():
    rng = np.random.default_rng(7)
    return ohlc_from_closes(C0 * np.cumprod(1 + rng.normal(0.0005, 0.007, N)))


def random_walk_mean_s(k: int = 20) -> float:
    vals = []
    for seed in range(100, 100 + k):
        rng = np.random.default_rng(seed)
        c = C0 * np.cumprod(1 + rng.normal(0.0005, 0.007, N))
        h, lo, cc = ohlc_from_closes(c)
        vals.append(month_features(h, lo, cc, C0)[3])
    return float(np.mean(vals))


@case("mixed")
def _():
    a = C0 * 1.005 ** t(11)
    tail = np.where(t(10) % 2 == 0, 1.015, 0.985)
    c = np.concatenate([a, a[-1] * np.cumprod(tail)])
    return ohlc_from_closes(c)


def main() -> int:
    res = {}
    for name, fn in CASES.items():
        h, lo, c = fn()
        res[name] = month_features(h, lo, c, C0)
    print(f"{'case':<18} {'T_range':>8} {'T_dir':>7} {'T_mono':>7} {'S_m':>6}")
    print("-" * 52)
    for name, (tr, td, tm, s) in sorted(res.items(), key=lambda kv: -kv[1][3]):
        print(f"{name:<18} {tr:>8.3f} {td:>7.3f} {tm:>7.3f} {s:>6.3f}")

    s = {k: v[3] for k, v in res.items()}
    td = {k: v[1] for k, v in res.items()}
    tm = {k: v[2] for k, v in res.items()}
    tr = {k: v[0] for k, v in res.items()}
    checks = [
        ("bull/bear direction-agnostic", abs(s["smooth_bull"] - s["smooth_bear"]) < 0.05),
        ("smooth trends score high", s["smooth_bull"] > 0.75 and s["smooth_bear"] > 0.75),
        ("staircase trends", s["staircase"] > 0.55),
        ("single shock: T_dir == 1/sqrt(N)", abs(td["shock"] - 1 / np.sqrt(N)) < 0.01),
        ("single shock not a perfect trend", s["shock"] < 0.60),
        ("reversal penalized", s["trend_reversal"] < s["smooth_bull"] - 0.15 and tm["trend_reversal"] < 0.6),
        ("chop scores low", s["volatile_sideways"] < 0.25 and s["quiet_sideways"] < 0.35),
        ("wicks hurt efficiency only", tr["trend_wicks"] < tr["smooth_bull"] and s["trend_wicks"] > 0.45),
        ("random walk not systematically trending", s["random_walk"] < 0.65 and random_walk_mean_s() < 0.4),
        ("clean trends beat chop+reversal",
         min(s["smooth_bull"], s["smooth_bear"], s["staircase"]) > max(s["volatile_sideways"], s["quiet_sideways"], s["trend_reversal"])),
        ("mixed sits in the middle", s["volatile_sideways"] < s["mixed"] < s["smooth_bull"]),
    ]
    ok_all = True
    for label, ok in checks:
        print(("PASS" if ok else "FAIL") + f": {label}")
        ok_all &= ok
    print("VALIDATION:", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
