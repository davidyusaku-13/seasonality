"""Causal one-month-ahead regime forecasting shared by backtest and live views."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.mixture import GaussianMixture

FEAT_PATH = Path("data/monthly_features.parquet")
FEATURES = ["t_range", "t_direction", "t_mono"]
MIN_N = 15
MIN_GMM_HISTORY = 48
MIN_FORECAST_HISTORY = 36
EPS = 1e-6
SEED = 0
CANDIDATES = [
    "global",
    "calendar",
    "calendar_calibrated",
    "persistence",
    "markov",
    "seasonal_markov",
    "logistic",
    "cyclic",
    "month_pool",
    "multiscale",
    "cyclic_multiscale",
    "month_pool_multiscale",
]
PROMOTABLE = [
    "global",
    "calendar_calibrated",
    "persistence",
    "markov",
    "seasonal_markov",
    "logistic",
    "cyclic",
    "month_pool",
    "multiscale",
    "cyclic_multiscale",
    "month_pool_multiscale",
]
RIDGE_MODELS = [
    "cyclic",
    "month_pool",
    "multiscale",
    "cyclic_multiscale",
    "month_pool_multiscale",
]


@dataclass(frozen=True)
class ForecastContext:
    target_year: int
    target_month: int
    q_prev: float
    q_prev2: float
    calendar_p: float
    global_p: float
    log_rv_1: float
    log_rv_3: float
    log_rv_12: float
    log_range_var_1: float
    jump_share_1: float
    semivar_imbalance_1: float


def _context_from_record(row: dict) -> ForecastContext:
    return ForecastContext(
        row["year"],
        row["month"],
        row["q_prev"],
        row["q_prev2"],
        row["calendar_p"],
        row["global_p"],
        row["log_rv_1"],
        row["log_rv_3"],
        row["log_rv_12"],
        row["log_range_var_1"],
        row["jump_share_1"],
        row["semivar_imbalance_1"],
    )


def _gmm(n_components: int) -> GaussianMixture:
    return GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        n_init=10,
        random_state=SEED,
        reg_covar=1e-3,
    )


def _next_month(year: int, month: int) -> tuple[int, int]:
    return (year + 1, 1) if month == 12 else (year, month + 1)


def _consecutive(a: dict, b: dict) -> bool:
    """True when b is the calendar month immediately after a."""
    return _next_month(int(a["year"]), int(a["month"])) == (
        int(b["year"]),
        int(b["month"]),
    )


def _clip_prob(value: float | np.ndarray) -> float | np.ndarray:
    return np.clip(value, EPS, 1 - EPS)


def _soft_brier(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return (p - q) ** 2 + q * (1 - q)


def _soft_log_loss(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = _clip_prob(p)
    return -(q * np.log(p) + (1 - q) * np.log(1 - p))


def _calibration_stats(p: np.ndarray, q: np.ndarray) -> tuple[float, float, float]:
    """Fractional-logistic calibration intercept/slope and three-bin ECE."""
    x = logit(_clip_prob(p))
    X = np.column_stack([np.ones(len(x)), x])

    def objective(beta: np.ndarray) -> float:
        fitted = _clip_prob(expit(X @ beta))
        return float(np.mean(_soft_log_loss(fitted, q)))

    result = minimize(
        objective,
        np.array([0.0, 1.0]),
        method="L-BFGS-B",
        bounds=[(-10.0, 10.0), (-10.0, 10.0)],
    )
    beta = result.x
    ece = 0.0
    for lo, hi in ((0.0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.01)):
        mask = (p >= lo) & (p < hi)
        if mask.any():
            ece += float(mask.mean()) * abs(float(p[mask].mean() - q[mask].mean()))
    return float(beta[0]), float(beta[1]), ece


def build_prequential_targets(feat: pl.DataFrame) -> list[dict]:
    """Assign q to month t using a GMM fitted strictly on months before t."""
    clean = feat.filter(pl.col("is_complete") & (pl.col("n") >= MIN_N)).sort(
        ["year", "month"]
    )
    rows = clean.to_dicts()
    for i, row in enumerate(rows):
        row["seq"] = i
        row["q_prequential"] = None
        row["component_crossed"] = None
        row["bic_advantage_2"] = None
        x_now = np.array([row[c] for c in FEATURES], dtype=float)
        if i < MIN_GMM_HISTORY or not np.isfinite(x_now).all():
            continue
        X = clean.slice(0, i).select(FEATURES).to_numpy()
        X = X[np.isfinite(X).all(axis=1)]
        if len(X) < MIN_GMM_HISTORY:
            continue
        g2 = _gmm(2).fit(X)
        g1 = _gmm(1).fit(X)
        scores = g2.means_.mean(axis=1)
        trend_idx = int(np.argmax(scores))
        other = 1 - trend_idx
        row["q_prequential"] = float(
            g2.predict_proba(x_now.reshape(1, -1))[0, trend_idx]
        )
        row["component_crossed"] = bool(
            np.any(g2.means_[trend_idx] <= g2.means_[other])
        )
        row["bic_advantage_2"] = float(g1.bic(X) - g2.bic(X))
    return rows


def _shrunk_mean(values: list[float]) -> float:
    return float((0.5 + sum(values)) / (len(values) + 1)) if values else 0.5


def _origin_multiscale(prequential: list[dict], origin_idx: int) -> dict | None:
    """Return state features ending at origin_idx, or None for a broken window."""
    if origin_idx < 11:
        return None
    window = prequential[origin_idx - 11 : origin_idx + 1]
    if any(not _consecutive(a, b) for a, b in pairwise(window)):
        return None
    required = [
        "realized_variance",
        "range_variance",
        "jump_share",
        "semivar_imbalance",
    ]
    values = np.array([[r[name] for name in required] for r in window], dtype=float)
    if not np.isfinite(values).all() or np.any(values[:, :2] <= 0):
        return None
    rv = values[:, 0]
    return {
        "log_rv_1": float(np.log(rv[-1])),
        "log_rv_3": float(np.log(rv[-3:].mean())),
        "log_rv_12": float(np.log(rv.mean())),
        "log_range_var_1": float(np.log(values[-1, 1])),
        "jump_share_1": float(values[-1, 2]),
        "semivar_imbalance_1": float(values[-1, 3]),
    }


def build_transition_records(prequential: list[dict]) -> list[dict]:
    """Create point-in-time supervised rows for q[t] from information at t-1."""
    out: list[dict] = []
    for i in range(1, len(prequential)):
        target, prev = prequential[i], prequential[i - 1]
        if not _consecutive(prev, target):
            continue
        qt, qp = target["q_prequential"], prev["q_prequential"]
        if qt is None or qp is None:
            continue
        multiscale = _origin_multiscale(prequential, i - 1)
        if multiscale is None:
            continue
        q2 = qp
        if i >= 2 and _consecutive(prequential[i - 2], prev):
            q2 = prequential[i - 2]["q_prequential"]
            q2 = qp if q2 is None else q2
        earlier = [r for r in prequential[:i] if r["q_prequential"] is not None]
        cal = [
            float(r["q_prequential"]) for r in earlier if r["month"] == target["month"]
        ]
        all_q = [float(r["q_prequential"]) for r in earlier]
        out.append(
            {
                "seq": int(target["seq"]),
                "year": int(target["year"]),
                "month": int(target["month"]),
                "ym": target["ym"],
                "origin_ym": prev["ym"],
                "q": float(qt),
                "q_prev": float(qp),
                "q_prev2": float(q2),
                "calendar_p": _shrunk_mean(cal),
                "global_p": _shrunk_mean(all_q),
                "component_crossed": bool(target["component_crossed"]),
                "bic_advantage_2": float(target["bic_advantage_2"]),
                **multiscale,
            }
        )
    return out


def _markov_params(rows: list[dict], month: int | None = None, strength: float = 0.0):
    qp = np.array([r["q_prev"] for r in rows], dtype=float)
    q = np.array([r["q"] for r in rows], dtype=float)
    den0, den1 = float(np.sum(1 - qp)), float(np.sum(qp))
    num0 = float(np.sum((1 - qp) * q))
    num1 = float(np.sum(qp * q))
    a0g = (0.5 + num0) / (1 + den0)
    a1g = (0.5 + num1) / (1 + den1)
    if month is None:
        return a0g, a1g
    mr = [r for r in rows if r["month"] == month]
    mqp = np.array([r["q_prev"] for r in mr], dtype=float)
    mq = np.array([r["q"] for r in mr], dtype=float)
    mden0, mden1 = float(np.sum(1 - mqp)), float(np.sum(mqp))
    mnum0 = float(np.sum((1 - mqp) * mq))
    mnum1 = float(np.sum(mqp * mq))
    return (
        (strength * a0g + mnum0) / (strength + mden0),
        (strength * a1g + mnum1) / (strength + mden1),
    )


def _markov_predict(rows: list[dict], q_prev: float, month=None, strength=0.0) -> float:
    a0, a1 = _markov_params(rows, month, strength)
    return float((1 - q_prev) * a0 + q_prev * a1)


def _logistic_x(calendar_p: float, q_prev: float, q_prev2: float) -> np.ndarray:
    return np.array(
        [
            float(logit(_clip_prob(calendar_p))),
            float(logit(_clip_prob(q_prev))),
            q_prev - q_prev2,
        ],
        dtype=float,
    )


def _fit_fractional_logistic(rows: list[dict], penalty: float):
    X0 = np.vstack(
        [_logistic_x(r["calendar_p"], r["q_prev"], r["q_prev2"]) for r in rows]
    )
    q = np.array([r["q"] for r in rows], dtype=float)
    mean = X0.mean(axis=0)
    scale = X0.std(axis=0)
    scale[scale < 1e-8] = 1.0
    X = np.column_stack([np.ones(len(X0)), (X0 - mean) / scale])
    beta = np.zeros(X.shape[1])
    ridge = np.diag([0.0, penalty, penalty, penalty])
    for _ in range(50):
        p = expit(X @ beta)
        w = np.maximum(p * (1 - p), 1e-6)
        grad = X.T @ (p - q) + ridge @ beta
        hess = X.T @ (X * w[:, None]) + ridge
        step = np.linalg.solve(hess, grad)
        beta -= step
        if float(np.max(np.abs(step))) < 1e-10:
            break
    return beta, mean, scale


def _fit_calendar_calibrator(rows: list[dict]) -> np.ndarray:
    p = np.array([r["calendar_p"] for r in rows], dtype=float)
    q = np.array([r["q"] for r in rows], dtype=float)
    x = logit(_clip_prob(p))

    def objective(beta: np.ndarray) -> float:
        fitted = _clip_prob(expit(beta[0] + beta[1] * x))
        return float(np.mean(_soft_log_loss(fitted, q)))

    result = minimize(
        objective,
        np.array([0.0, 1.0]),
        method="L-BFGS-B",
        bounds=[(-10.0, 10.0), (-10.0, 10.0)],
    )
    return result.x


def _calendar_calibrated_predict(beta: np.ndarray, calendar_p: float) -> float:
    return float(expit(beta[0] + beta[1] * logit(_clip_prob(calendar_p))))


def _logistic_predict(model, context: ForecastContext) -> float:
    beta, mean, scale = model
    x0 = _logistic_x(context.calendar_p, context.q_prev, context.q_prev2)
    x = np.r_[1.0, (x0 - mean) / scale]
    return float(expit(x @ beta))


def _cyclic_x(month: int) -> np.ndarray:
    angle = 2 * np.pi * (month - 1) / 12
    return np.array(
        [np.sin(angle), np.cos(angle), np.sin(2 * angle), np.cos(2 * angle)]
    )


def _month_pool_x(month: int) -> np.ndarray:
    effects = np.full(12, -1 / 12)
    effects[month - 1] += 1
    return effects


def _multiscale_x(context: ForecastContext) -> np.ndarray:
    return np.array(
        [
            float(logit(_clip_prob(context.q_prev))),
            context.q_prev - context.q_prev2,
            context.log_rv_1,
            context.log_rv_3,
            context.log_rv_12,
            context.log_range_var_1,
            context.jump_share_1,
            context.semivar_imbalance_1,
        ]
    )


def _ridge_x(model: str, context: ForecastContext) -> np.ndarray:
    cyclic = _cyclic_x(context.target_month)
    month_pool = _month_pool_x(context.target_month)
    multiscale = _multiscale_x(context)
    if model == "cyclic":
        return cyclic
    if model == "month_pool":
        return month_pool
    if model == "multiscale":
        return multiscale
    if model == "cyclic_multiscale":
        return np.r_[cyclic, multiscale]
    if model == "month_pool_multiscale":
        return np.r_[month_pool, multiscale]
    raise ValueError(f"unknown ridge model: {model}")


def _fit_ridge_model(rows: list[dict], model: str, penalty: float):
    X0 = np.vstack([_ridge_x(model, _context_from_record(r)) for r in rows])
    q = np.array([r["q"] for r in rows], dtype=float)
    mean = X0.mean(axis=0)
    scale = X0.std(axis=0)
    scale[scale < 1e-8] = 1.0
    X = np.column_stack([np.ones(len(X0)), (X0 - mean) / scale])
    beta = np.zeros(X.shape[1])
    ridge = np.diag(np.r_[0.0, np.full(X0.shape[1], penalty)])
    for _ in range(50):
        p = expit(X @ beta)
        w = np.maximum(p * (1 - p), 1e-6)
        grad = X.T @ (p - q) + ridge @ beta
        hess = X.T @ (X * w[:, None]) + ridge
        step = np.linalg.solve(hess, grad)
        beta -= step
        if float(np.max(np.abs(step))) < 1e-10:
            break
    return beta, mean, scale


def _ridge_predict(fitted, model: str, context: ForecastContext) -> float:
    beta, mean, scale = fitted
    x0 = _ridge_x(model, context)
    return float(expit(np.r_[1.0, (x0 - mean) / scale] @ beta))


def _inner_origins(n: int) -> list[int]:
    points = list(range(24, n, 6))
    if n > 24 and (not points or points[-1] != n - 1):
        points.append(n - 1)
    return sorted(set(points))


def _choose_strength(rows: list[dict]) -> float:
    grid = [2.0, 6.0, 12.0, 24.0]
    scores = []
    for strength in grid:
        losses = []
        for i in _inner_origins(len(rows)):
            r = rows[i]
            p = _markov_predict(rows[:i], r["q_prev"], r["month"], strength)
            losses.append((p - r["q"]) ** 2)
        scores.append(float(np.mean(losses)) if losses else np.inf)
    return grid[int(np.argmin(scores))]


def _choose_penalty(rows: list[dict]) -> float:
    grid = [0.01, 0.1, 1.0, 10.0, 100.0]
    scores = []
    for penalty in grid:
        losses = []
        for i in _inner_origins(len(rows)):
            model = _fit_fractional_logistic(rows[:i], penalty)
            r = rows[i]
            ctx = _context_from_record(r)
            losses.append((_logistic_predict(model, ctx) - r["q"]) ** 2)
        scores.append(float(np.mean(losses)) if losses else np.inf)
    return grid[int(np.argmin(scores))]


def _choose_ridge_penalty(rows: list[dict], model: str) -> float:
    grid = [0.01, 0.1, 1.0, 10.0, 100.0]
    scores = []
    for penalty in grid:
        losses = []
        for i in _inner_origins(len(rows)):
            fitted = _fit_ridge_model(rows[:i], model, penalty)
            r = rows[i]
            p = _ridge_predict(fitted, model, _context_from_record(r))
            losses.append((p - r["q"]) ** 2)
        scores.append(float(np.mean(losses)) if losses else np.inf)
    return grid[int(np.argmin(scores))]


def predict_candidates(rows: list[dict], context: ForecastContext) -> tuple[dict, dict]:
    """Fit every candidate on rows and predict the supplied next month."""
    strength = _choose_strength(rows)
    penalty = _choose_penalty(rows)
    logistic_model = _fit_fractional_logistic(rows, penalty)
    calendar_calibrator = _fit_calendar_calibrator(rows)
    ridge_penalties = {
        model: _choose_ridge_penalty(rows, model) for model in RIDGE_MODELS
    }
    ridge_fits = {
        model: _fit_ridge_model(rows, model, ridge_penalties[model])
        for model in RIDGE_MODELS
    }
    pred = {
        "global": context.global_p,
        "calendar": context.calendar_p,
        "calendar_calibrated": _calendar_calibrated_predict(
            calendar_calibrator, context.calendar_p
        ),
        "persistence": context.q_prev,
        "markov": _markov_predict(rows, context.q_prev),
        "seasonal_markov": _markov_predict(
            rows, context.q_prev, context.target_month, strength
        ),
        "logistic": _logistic_predict(logistic_model, context),
        **{
            model: _ridge_predict(ridge_fits[model], model, context)
            for model in RIDGE_MODELS
        },
    }
    return {k: float(_clip_prob(v)) for k, v in pred.items()}, {
        "seasonal_strength": strength,
        "logistic_penalty": penalty,
        **{f"{model}_penalty": value for model, value in ridge_penalties.items()},
    }


def run_backtest(prequential: list[dict]) -> tuple[list[dict], list[dict], str, bool]:
    records = build_transition_records(prequential)
    predictions: list[dict] = []
    for i in range(MIN_FORECAST_HISTORY, len(records)):
        r = records[i]
        hist = records[:i]
        ctx = _context_from_record(r)
        pred, tuning = predict_candidates(hist, ctx)
        out = {**r, **{f"p_{k}": v for k, v in pred.items()}, **tuning}
        predictions.append(out)

    comparison, champion, regime_ok = select_champion(predictions)
    return predictions, comparison, champion, regime_ok


def select_champion(predictions: list[dict]) -> tuple[list[dict], str, bool]:
    """Select from a 95% block-bootstrap model confidence set."""
    crossed_rate = float(np.mean([r["component_crossed"] for r in predictions]))
    bic_support = float(np.mean([r["bic_advantage_2"] > 0 for r in predictions]))
    regime_ok = crossed_rate <= 0.05 and bic_support >= 0.80
    comparison = compare_models(predictions)
    included = model_confidence_set(predictions)
    for row in comparison:
        row["mcs_included"] = row["model"] in included
    passing = [r for r in comparison if r["model"] in PROMOTABLE and r["mcs_included"]]
    champion = (
        min(passing, key=lambda r: (r["brier"], PROMOTABLE.index(r["model"])))["model"]
        if passing and "calendar" not in included and regime_ok
        else "calendar"
    )
    return comparison, champion, regime_ok


def model_confidence_set(
    rows: list[dict], alpha: float = 0.05, n_boot: int = 2000
) -> set[str]:
    """Tmax elimination using deterministic calendar-year block bootstraps."""
    years = sorted({r["year"] for r in rows})
    year_indices = {
        year: np.array([i for i, row in enumerate(rows) if row["year"] == year])
        for year in years
    }
    q = np.array([r["q"] for r in rows], dtype=float)
    losses = {
        model: _soft_brier(np.array([r[f"p_{model}"] for r in rows], dtype=float), q)
        for model in CANDIDATES
    }
    active = list(CANDIDATES)
    rng = np.random.default_rng(SEED)
    samples = [
        np.concatenate(
            [year_indices[int(y)] for y in rng.choice(years, len(years), replace=True)]
        )
        for _ in range(n_boot)
    ]
    while len(active) > 1:
        matrix = np.column_stack([losses[model] for model in active])
        relative = matrix - matrix.mean(axis=1, keepdims=True)
        observed_means = relative.mean(axis=0)
        boot_means = np.vstack([relative[idx].mean(axis=0) for idx in samples])
        se = boot_means.std(axis=0, ddof=1)
        se[se < 1e-12] = np.inf
        observed_t = observed_means / se
        boot_t = (boot_means - observed_means) / se
        statistic = float(np.max(observed_t))
        bootstrap_statistics = np.max(boot_t, axis=1)
        p_value = (1 + int(np.sum(bootstrap_statistics >= statistic))) / (n_boot + 1)
        if p_value >= alpha:
            break
        del active[int(np.argmax(observed_t))]
    return set(active)


def _year_bootstrap_diff(
    rows: list[dict], model: str, n_boot: int = 2000
) -> tuple[float, float]:
    years = sorted({r["year"] for r in rows})
    by_year = {y: [r for r in rows if r["year"] == y] for y in years}
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(n_boot):
        sampled = rng.choice(years, size=len(years), replace=True)
        rr = [r for y in sampled for r in by_year[int(y)]]
        q = np.array([r["q"] for r in rr])
        pm = np.array([r[f"p_{model}"] for r in rr])
        pc = np.array([r["p_calendar"] for r in rr])
        vals.append(float(np.mean(_soft_brier(pm, q) - _soft_brier(pc, q))))
    return tuple(float(v) for v in np.quantile(vals, [0.025, 0.975]))


def compare_models(rows: list[dict]) -> list[dict]:
    q = np.array([r["q"] for r in rows], dtype=float)
    out = []
    for model in CANDIDATES:
        p = np.array([r[f"p_{model}"] for r in rows], dtype=float)
        lo, hi = _year_bootstrap_diff(rows, model)
        cal_intercept, cal_slope, ece = _calibration_stats(p, q)
        out.append(
            {
                "model": model,
                "n": len(rows),
                "brier": float(np.mean(_soft_brier(p, q))),
                "log_loss": float(np.mean(_soft_log_loss(p, q))),
                "hard_agreement": float(np.mean((p >= 0.5) == (q >= 0.5))),
                "mean_pred": float(p.mean()),
                "mean_realized": float(q.mean()),
                "calibration_intercept": cal_intercept,
                "calibration_slope": cal_slope,
                "calibration_ece_3bin": ece,
                "brier_diff_vs_calendar": float(
                    np.mean(
                        _soft_brier(p, q)
                        - _soft_brier(np.array([r["p_calendar"] for r in rows]), q)
                    )
                ),
                "brier_diff_ci95_lo": lo,
                "brier_diff_ci95_hi": hi,
            }
        )
    return out


def live_context(prequential: list[dict]) -> tuple[list[dict], ForecastContext, dict]:
    usable = [r for r in prequential if r["q_prequential"] is not None]
    if len(usable) < 2:
        raise ValueError("insufficient prequential regime history")
    latest, prior = usable[-1], usable[-2]
    if not _consecutive(prior, latest):
        raise ValueError("latest valid months are not consecutive")
    ty, tm = _next_month(latest["year"], latest["month"])
    cal_q = [float(r["q_prequential"]) for r in usable if r["month"] == tm]
    all_q = [float(r["q_prequential"]) for r in usable]
    latest_idx = int(latest["seq"])
    multiscale = _origin_multiscale(prequential, latest_idx)
    if multiscale is None:
        raise ValueError("insufficient consecutive months for multiscale live features")
    context = ForecastContext(
        ty,
        tm,
        float(latest["q_prequential"]),
        float(prior["q_prequential"]),
        _shrunk_mean(cal_q),
        _shrunk_mean(all_q),
        **multiscale,
    )
    return build_transition_records(prequential), context, latest


def live_interval(
    rows: list[dict],
    context: ForecastContext,
    model: str,
    tuning: dict,
    n_boot: int = 2000,
) -> tuple[float, float]:
    years = sorted({r["year"] for r in rows})
    by_year = {y: [r for r in rows if r["year"] == y] for y in years}
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(n_boot):
        sampled = rng.choice(years, size=len(years), replace=True)
        rr = [r for y in sampled for r in by_year[int(y)]]
        boot_context = ForecastContext(
            context.target_year,
            context.target_month,
            context.q_prev,
            context.q_prev2,
            _shrunk_mean([r["q"] for r in rr if r["month"] == context.target_month]),
            _shrunk_mean([r["q"] for r in rr]),
            context.log_rv_1,
            context.log_rv_3,
            context.log_rv_12,
            context.log_range_var_1,
            context.jump_share_1,
            context.semivar_imbalance_1,
        )
        if model == "calendar":
            vals.append(boot_context.calendar_p)
        elif model == "calendar_calibrated":
            calibrator = _fit_calendar_calibrator(rr)
            vals.append(
                _calendar_calibrated_predict(calibrator, boot_context.calendar_p)
            )
        elif model == "global":
            vals.append(boot_context.global_p)
        elif model == "persistence":
            vals.append(context.q_prev)
        elif model == "markov":
            vals.append(_markov_predict(rr, context.q_prev))
        elif model == "seasonal_markov":
            vals.append(
                _markov_predict(
                    rr,
                    context.q_prev,
                    context.target_month,
                    tuning["seasonal_strength"],
                )
            )
        elif model == "logistic":
            fitted = _fit_fractional_logistic(rr, tuning["logistic_penalty"])
            vals.append(_logistic_predict(fitted, boot_context))
        else:
            fitted = _fit_ridge_model(rr, model, tuning[f"{model}_penalty"])
            vals.append(_ridge_predict(fitted, model, boot_context))
    return tuple(float(v) for v in np.quantile(vals, [0.025, 0.975]))
