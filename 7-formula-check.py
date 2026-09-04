"""Formula correctness checks (FORMULA.md steps 1-8) before any model learning.

Covers: exact hand-computed values, scale/time-reversal invariants,
edge cases (sec 17), N-bias and component redundancy on real data,
noise monotonicity, and a GBM null distribution for chance level.
Exit 0 iff all checks pass.
"""

import importlib.util
from itertools import pairwise
from pathlib import Path

import numpy as np
import polars as pl

SPEC = Path(__file__).with_name("2-seasonality.py")
_spec = importlib.util.spec_from_file_location("seasonality2", SPEC)
s2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s2)
mf = s2.month_features

N = 21
C0 = 3000.0
checks: list[tuple[str, bool, str]] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    checks.append((label, ok, detail))
    print(
        ("PASS" if ok else "FAIL") + f": {label}" + (f" ({detail})" if detail else "")
    )


def ohlc(c: np.ndarray, wick: float = 0.0, c0: float = C0):
    c = np.asarray(c, dtype=float)
    prev = np.concatenate([[c0], c[:-1]])
    h = np.maximum(c, prev) * (1 + wick)
    lo = np.minimum(c, prev) * (1 - wick)
    return h, lo, c


# 1. perfect 2-step trend -> all ones (hand-computed: TR sum=Rm=21, r equal, tau=1)
tr, td, tm, s = mf(*ohlc(np.array([110.0, 121.0]), 0.0, 100.0), 100.0)
check(
    "perfect trend scores 1",
    (tr, td, tm, s) == (1.0, 1.0, 1.0, 1.0),
    f"{tr, td, tm, s}",
)

# 2. round-trip 100->101->100, no wicks: TR sum=2, Rm=1 -> T_range = 1-ln2/ln2 = 0
tr, td, tm, s = mf(*ohlc(np.array([101.0, 100.0]), 0.0, 100.0), 100.0)
check("round-trip T_range is 0", tr == 0.0 and s == 0.0, f"T_range={tr} S={s}")

# 3. flat month -> invalid / degenerate (FORMULA sec 17.2)
tr, td, tm, s = mf(*ohlc(np.full(10, C0)), C0)
check("flat month is invalid", bool(np.isnan([tr, td, tm, s]).all()))

# Incomplete months must not enter calendar averages, even when they have N>=15.
fake = pl.DataFrame(
    {
        "month": [1, 1],
        "n": [20, 20],
        "is_complete": [True, False],
        "q_trend": [0.2, 0.9],
    }
)
jan = s2.calendar_table(fake).filter(pl.col("month") == 1).to_dicts()[0]
check(
    "incomplete N>=15 month excluded from calendar mean",
    jan["n"] == 1 and abs(jan["p_trend_raw"] - 0.2) < 1e-12,
)

# 4. N=1 -> zeros, no div-by-zero
tr, td, tm, s = mf(np.array([101.0]), np.array([99.0]), np.array([100.5]), 100.0)
check("N=1 is 0, no crash", (tr, td, tm, s) == (0.0, 0.0, 0.0, 0.0))

try:
    s2.clip_unit(1.01, "synthetic")
    rejected_bad_bound = False
except ValueError:
    rejected_bad_bound = True
check("material bound violation rejected", rejected_bad_bound)

# 5. scale invariance x100
rng = np.random.default_rng(3)
c = C0 * np.cumprod(1 + rng.normal(0.001, 0.008, N))
a = mf(*ohlc(c, 0.002), C0)
b = mf(*(v * 100 for v in ohlc(c, 0.002)), C0 * 100)
check(
    "scale x100 invariant",
    np.allclose(a, b, atol=1e-9),
    f"maxdiff={max(abs(x - y) for x, y in zip(a, b)):.2e}",
)

origin_a = s2.forecast_origin_features(*ohlc(c, 0.002), C0)
origin_b = s2.forecast_origin_features(*(v * 100 for v in ohlc(c, 0.002)), C0 * 100)
check(
    "forecast-origin features scale invariant",
    np.allclose(origin_a, origin_b, atol=1e-12),
    f"maxdiff={max(abs(x - y) for x, y in zip(origin_a, origin_b)):.2e}",
)

shock_c = np.full(N, C0)
shock_c[5:] = C0 * 1.1
shock_origin = s2.forecast_origin_features(*ohlc(shock_c), C0)
check(
    "single shock has unit jump share and semivariance imbalance",
    np.isclose(shock_origin[2], 1.0) and np.isclose(shock_origin[3], 1.0),
    f"jump={shock_origin[2]:.3f} imbalance={shock_origin[3]:.3f}",
)

# 6. time-reversal invariance (proper path reversal incl. H/L and C0)
h, lo, c = ohlc(c, 0.002)
p = np.concatenate([[C0], c])
q = p[::-1]
ar = mf(h[::-1].copy(), lo[::-1].copy(), q[1:].copy(), float(q[0]))
check(
    "time reversal invariant",
    np.allclose(a, ar, atol=1e-9),
    f"maxdiff={max(abs(x - y) for x, y in zip(a, ar)):.2e}",
)

# 7. noise monotonicity: same drift, rising noise -> falling mean S
base = C0 * 1.005 ** np.arange(1, N + 1)
means = []
for sig in (0.0, 0.002, 0.005, 0.01, 0.02):
    vals = []
    for seed in range(10):
        r = np.random.default_rng(1000 + seed)
        cc = base * np.exp(
            r.normal(0, sig, N).cumsum() - 0.5 * sig**2 * np.arange(1, N + 1)
        )
        prev = np.concatenate([[C0], cc[:-1]])
        vals.append(
            mf(np.maximum(cc, prev) * 1.0005, np.minimum(cc, prev) * 0.9995, cc, C0)[3]
        )
    means.append(float(np.mean(vals)))
check(
    "noise lowers S monotonically",
    all(x > y for x, y in pairwise(means)),
    f"S={[round(v, 3) for v in means]}",
)

# 8. real data: no nan, N-bias, component redundancy
feat = pl.read_parquet("data/monthly_features.parquet").filter(pl.col("n") >= 15)
X = feat.select(
    [
        "t_range",
        "t_direction",
        "t_mono",
        "s",
        "n",
        "realized_variance",
        "range_variance",
        "jump_share",
        "semivar_imbalance",
    ]
).to_numpy()
check("no nan in 230 full months", bool(np.isfinite(X).all()))
cs = float(np.corrcoef(X[:, 4], X[:, 3])[0, 1])
check("S_m nearly N-independent", abs(cs) < 0.15, f"corr(S,N)={cs:+.3f}")
C = np.corrcoef(X[:, :3].T)
mx = float(max(C[0, 1], C[0, 2], C[1, 2]))
check(
    "components not redundant",
    mx < 0.9,
    f"corr={np.round(C, 2).tolist()} max_offdiag={mx:.2f}",
)

# 9. GBM null: chance level for S_m
real = (
    pl.read_parquet("data/xauusd_d1.parquet").filter(~pl.col("is_filled")).sort("time")
)
cc = np.array(real["close"].to_list())
sig = float(np.std(np.diff(np.log(cc))))
null = []
for seed in range(500):
    r = np.random.default_rng(seed)
    sim = cc[0] * np.exp(np.cumsum(r.normal(0, sig, N)))
    prev = np.concatenate([[cc[0]], sim[:-1]])
    null.append(
        mf(np.maximum(sim, prev) * 1.0005, np.minimum(sim, prev) * 0.9995, sim, cc[0])[
            3
        ]
    )
null = np.array(null)
print(
    f"null GBM: sigma_daily={sig:.4f} mean_S={null.mean():.3f} "
    f"p5={np.percentile(null, 5):.3f} p50={np.percentile(null, 50):.3f} p95={np.percentile(null, 95):.3f}"
)
check(
    "null sits well below trend scores",
    null.mean() < 0.4 and np.percentile(null, 95) < 0.65,
)

n_fail = sum(not ok for _, ok, _ in checks)
print("FORMULA-CHECK:", "PASS" if n_fail == 0 else f"FAIL ({n_fail})")
raise SystemExit(1 if n_fail else 0)
