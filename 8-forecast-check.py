"""Causality, determinism, and forecast-interface checks."""

import numpy as np
import polars as pl

import forecasting as fc

checks: list[tuple[str, bool, str]] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    checks.append((label, ok, detail))
    print(
        ("PASS" if ok else "FAIL") + f": {label}" + (f" ({detail})" if detail else "")
    )


feat = pl.read_parquet(fc.FEAT_PATH).sort(["year", "month"])
base = fc.build_prequential_targets(feat)
again = fc.build_prequential_targets(feat)

qa = [r["q_prequential"] for r in base]
qb = [r["q_prequential"] for r in again]
check(
    "prequential labels deterministic",
    all(
        (a is None and b is None) or np.isclose(a, b, atol=1e-12)
        for a, b in zip(qa, qb, strict=True)
    ),
)
check(
    "minimum GMM history enforced",
    all(q is None for q in qa[: fc.MIN_GMM_HISTORY])
    and qa[fc.MIN_GMM_HISTORY] is not None,
)

# Alter the latest completed month's features. Earlier labels must be bit-identical.
latest = base[-1]
changed_names = fc.FEATURES + [
    "realized_variance",
    "range_variance",
    "jump_share",
    "semivar_imbalance",
]
changed = feat.with_columns(
    [
        pl.when(
            (pl.col("year") == latest["year"]) & (pl.col("month") == latest["month"])
        )
        .then(pl.lit(0.01))
        .otherwise(pl.col(name))
        .alias(name)
        for name in changed_names
    ]
)
altered = fc.build_prequential_targets(changed)
unchanged_past = all(
    (a["q_prequential"] is None and b["q_prequential"] is None)
    or np.isclose(a["q_prequential"], b["q_prequential"], atol=1e-12)
    for a, b in zip(base[:-1], altered[:-1], strict=True)
)
check("future observation cannot alter earlier labels", unchanged_past)

records = fc.build_transition_records(base)
check(
    "transition rows are consecutive months",
    all(
        fc._next_month(int(r["origin_ym"][:4]), int(r["origin_ym"][5:]))
        == (r["year"], r["month"])
        for r in records
    ),
)

predictions, comparison, champion, regime_ok = fc.run_backtest(base)
altered_predictions, _, _, _ = fc.run_backtest(altered)
past_forecasts_unchanged = all(
    all(
        np.isclose(a[f"p_{model}"], b[f"p_{model}"], atol=1e-12)
        for model in fc.CANDIDATES
    )
    for a, b in zip(predictions[:-1], altered_predictions[:-1], strict=True)
)
check("future observation cannot alter earlier forecasts", past_forecasts_unchanged)
base_mcs = fc.model_confidence_set(predictions[:-1], n_boot=500)
altered_mcs = fc.model_confidence_set(altered_predictions[:-1], n_boot=500)
check("future observation cannot alter earlier MCS membership", base_mcs == altered_mcs)
probs = np.array(
    [[r[f"p_{model}"] for model in fc.CANDIDATES] for r in predictions], dtype=float
)
check(
    "all out-of-sample probabilities bounded",
    bool(np.isfinite(probs).all() and ((probs > 0) & (probs < 1)).all()),
    f"n={len(predictions)}",
)
check(
    "regime component ordering stable",
    regime_ok,
    f"crossed={sum(r['component_crossed'] for r in predictions)}",
)
check(
    "promotion rule returns declared candidate",
    champion in fc.CANDIDATES and sum(r["model"] == champion for r in comparison) == 1,
    champion,
)

# Soft transition counts should preserve obvious persistence.
persistent = []
for year in range(20):
    persistent.extend(
        [
            {"year": year, "month": 1, "q_prev": 0.99, "q": 0.99},
            {"year": year, "month": 2, "q_prev": 0.01, "q": 0.01},
        ]
    )
high = fc._markov_predict(persistent, 0.99)
low = fc._markov_predict(persistent, 0.01)
check("soft Markov model detects persistence", high > 0.9 and low < 0.1)

check(
    "cyclic and pooled-month encodings are centered",
    abs(float(fc._cyclic_x(1)[0])) < 1e-12
    and abs(float(fc._month_pool_x(1).sum())) < 1e-12,
)
state_names = [
    "log_rv_1",
    "log_rv_3",
    "log_rv_12",
    "log_range_var_1",
    "jump_share_1",
    "semivar_imbalance_1",
]
check(
    "multiscale rolling features finite",
    all(np.isfinite([r[name] for name in state_names]).all() for r in records),
    f"n={len(records)}",
)

rng = np.random.default_rng(44)
separated = []
equal = []
for i in range(120):
    q = float(rng.uniform(0.05, 0.95))
    common = {"year": 2000 + i // 12, "q": q}
    separated.append(
        {
            **common,
            **{
                f"p_{model}": (1 - q if model == "calendar" else q)
                for model in fc.CANDIDATES
            },
        }
    )
    equal.append({**common, **{f"p_{model}": 0.5 for model in fc.CANDIDATES}})
separated_mcs = fc.model_confidence_set(separated, n_boot=500)
equal_mcs = fc.model_confidence_set(equal, n_boot=500)
check(
    "MCS excludes clearly inferior calendar",
    "calendar" not in separated_mcs and len(separated_mcs) == len(fc.CANDIDATES) - 1,
)
check("MCS retains equal models", equal_mcs == set(fc.CANDIDATES))
check(
    "MCS deterministic",
    separated_mcs == fc.model_confidence_set(separated, n_boot=500),
)


def synthetic_context(month: int, signal: float = 0.0) -> dict:
    return {
        "year": 2000,
        "month": month,
        "q_prev": 0.5,
        "q_prev2": 0.5,
        "calendar_p": 0.5,
        "global_p": 0.5,
        "log_rv_1": signal,
        "log_rv_3": signal,
        "log_rv_12": signal,
        "log_range_var_1": signal,
        "jump_share_1": 0.2,
        "semivar_imbalance_1": 0.2,
    }


cyclic_rows = []
month_rows = []
multiscale_rows = []
for year in range(10):
    for month in range(1, 13):
        angle = 2 * np.pi * (month - 1) / 12
        cyclic_rows.append(
            {
                **synthetic_context(month),
                "year": 2000 + year,
                "q": 0.5 + 0.35 * np.sin(angle),
            }
        )
        month_rows.append(
            {
                **synthetic_context(month),
                "year": 2000 + year,
                "q": 0.85 if month == 1 else 0.35,
            }
        )
        signal = 1.0 if month % 2 else -1.0
        multiscale_rows.append(
            {
                **synthetic_context(month, signal),
                "year": 2000 + year,
                "q": 0.8 if signal > 0 else 0.2,
            }
        )

cyclic_fit = fc._fit_ridge_model(cyclic_rows, "cyclic", 0.1)
cyclic_hi = fc._ridge_predict(
    cyclic_fit, "cyclic", fc._context_from_record(synthetic_context(4))
)
cyclic_lo = fc._ridge_predict(
    cyclic_fit, "cyclic", fc._context_from_record(synthetic_context(10))
)
check(
    "cyclic model recovers synthetic seasonality", cyclic_hi > 0.75 and cyclic_lo < 0.25
)

month_fit = fc._fit_ridge_model(month_rows, "month_pool", 0.1)
jan = fc._ridge_predict(
    month_fit, "month_pool", fc._context_from_record(synthetic_context(1))
)
feb = fc._ridge_predict(
    month_fit, "month_pool", fc._context_from_record(synthetic_context(2))
)
check("pooled-month model recovers sparse month effect", jan > 0.75 and feb < 0.45)

multi_fit = fc._fit_ridge_model(multiscale_rows, "multiscale", 0.1)
multi_hi = fc._ridge_predict(
    multi_fit, "multiscale", fc._context_from_record(synthetic_context(1, 1.0))
)
multi_lo = fc._ridge_predict(
    multi_fit, "multiscale", fc._context_from_record(synthetic_context(2, -1.0))
)
check("multiscale model recovers synthetic state", multi_hi > 0.7 and multi_lo < 0.3)

n_fail = sum(not ok for _, ok, _ in checks)
print("FORECAST-CHECK:", "PASS" if n_fail == 0 else f"FAIL ({n_fail})")
raise SystemExit(1 if n_fail else 0)
