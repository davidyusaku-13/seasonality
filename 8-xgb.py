"""Partial-month scorer: first-k-days data -> final [T_range, T_dir, T_mono].

S_m derives by geometric mean and q_m via the saved GMM artifact, exactly
as for scored months, so nowcasts live on the same scales. Trains one
XGB regressor per component (MultiOutputRegressor) on complete months
(N>=15): for each month and each k in 3..min(20, N-1), features come
strictly from the first k days plus the prior close C0. Time split
(train years < 2021, test >= 2021). Baselines: partial components/S_m
as predictors, and the GMM posterior of the partial month for q.

Outputs: models/xgb_partial_multi.joblib, models/xgb_partial_metrics.json
"""

import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor

SPEC = Path(__file__).with_name("2-seasonality.py")
_spec = importlib.util.spec_from_file_location("seasonality2", SPEC)
s2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s2)

SRC = Path("data/xauusd_d1.parquet")
FEAT = Path("data/monthly_features.parquet")
GMM_ART = Path("models/gmm_regime.joblib")
MODEL_OUT = Path("models/xgb_partial_multi.joblib")
MIN_N = 15
K_MIN, K_MAX = 3, 20
SPLIT_YEAR = 2021
TARGETS = ["t_range", "t_direction", "t_mono"]

FEATURES = [
    "k",
    "pT_range",
    "pT_dir",
    "pT_mono",
    "pS",
    "pret_total",
    "pvol",
    "frac_up",
    "max_up",
    "max_dn",
]


def build_partial_features(
    h: np.ndarray, low: np.ndarray, c: np.ndarray, c0: float, k: int
) -> list[float]:
    """Features from the first k days only (plus prior close C0)."""
    hk, lk, ck = h[:k], low[:k], c[:k]
    p_tr, p_td, p_tm, p_s = s2.month_features(hk, lk, ck, c0)
    prev = np.concatenate([[c0], ck[:-1]])
    r = np.log(ck / prev)
    return [
        float(k),
        p_tr,
        p_td,
        p_tm,
        p_s,
        float(np.log(ck[-1] / c0)),
        float(np.std(r)),
        float(np.mean(r > 0)),
        float(np.max(ck) / c0 - 1),
        float(np.min(ck) / c0 - 1),
    ]


def geomean_row(t: np.ndarray) -> float:
    t = np.clip(t, 0, 1)
    return 0.0 if t.min() <= 0 else float(np.prod(t) ** (1 / 3))


def month_rows(real: pl.DataFrame) -> list[dict]:
    """Per-month OHLC arrays with prior close (real bars only)."""
    times = real["time"].to_list()
    closes_all = [float(v) for v in real["close"].to_list()]
    out = []
    for y, m in sorted({(d.year, d.month) for d in real["date"].to_list()}):
        mb = real.filter(
            (pl.col("date").dt.year() == y) & (pl.col("date").dt.month() == m)
        ).sort("time")
        idx = times.index(mb["time"].min())
        c0 = closes_all[idx - 1] if idx > 0 else float(mb["open"].to_list()[0])
        out.append(
            {
                "year": y,
                "month": m,
                "ym": f"{y}-{m:02d}",
                "h": np.array(mb["high"].to_list(), dtype=float),
                "low": np.array(mb["low"].to_list(), dtype=float),
                "c": np.array(mb["close"].to_list(), dtype=float),
                "c0": float(c0),
            }
        )
    return out


def main() -> None:
    real = pl.read_parquet(SRC).filter(~pl.col("is_filled")).sort("time")
    feat = {r["ym"]: r for r in pl.read_parquet(FEAT).to_dicts()}
    art = joblib.load(GMM_ART)
    gmm_full, tidx_full = art["gmm"], art["trend_idx"]
    months = month_rows(real)

    Xs, Ys, ks, yms = [], [], [], []
    for mo in months:
        n = len(mo["c"])
        if n < MIN_N or mo["ym"] not in feat:
            continue
        for k in range(K_MIN, min(K_MAX, n - 1) + 1):
            Xs.append(build_partial_features(mo["h"], mo["low"], mo["c"], mo["c0"], k))
            fr = feat[mo["ym"]]
            Ys.append([fr[t] for t in TARGETS])
            ks.append(k)
            yms.append(mo["ym"])
    X, Y, ks, yms = (np.array(a) for a in (Xs, Ys, ks, yms))
    tr = yms < f"{SPLIT_YEAR}-01"
    print(f"rows={len(Y)} train={int(tr.sum())} test={int((~tr).sum())}")

    base = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        seed=0,
        n_jobs=1,
    )
    model = MultiOutputRegressor(base)
    model.fit(X[tr], Y[tr])
    P = model.predict(X[te := ~tr])

    # realized q from a train-window-only GMM (no lookahead in eval labels)
    from sklearn.mixture import GaussianMixture

    g = GaussianMixture(
        n_components=2,
        covariance_type="full",
        n_init=10,
        random_state=0,
        reg_covar=1e-3,
    )
    Xtr_full = np.array(
        [
            [feat[r["ym"]][t] for t in TARGETS]
            for r in pl.read_parquet(FEAT)
            .filter((pl.col("year") < SPLIT_YEAR) & (pl.col("n") >= MIN_N))
            .to_dicts()
        ]
    )
    g.fit(Xtr_full)
    tidx_tr = int(np.argsort(g.means_.mean(axis=1))[1])
    q_real = g.predict_proba(Y[te])[:, tidx_tr]

    s_hat = np.array([geomean_row(p) for p in P])
    s_true = np.array([geomean_row(t) for t in Y[te]])
    q_hat = gmm_full.predict_proba(np.clip(P, 0, 1))[:, tidx_full]
    naive_s = X[te][:, FEATURES.index("pS")]
    naive_q = gmm_full.predict_proba(np.clip(X[te][:, 1:4], 0, 1))[:, tidx_full]

    print("--- per-component test MAE (xgb vs partial-as-predictor) ---")
    for j, t in enumerate(TARGETS):
        print(
            f"{t:<12} xgb={np.abs(P[:, j] - Y[te][:, j]).mean():.4f} "
            f"naive={np.abs(X[te][:, 1 + j] - Y[te][:, j]).mean():.4f}"
        )
    ex, en = np.abs(s_hat - s_true), np.abs(naive_s - s_true)
    print(
        f"S_m: MAE xgb={ex.mean():.4f} naive={en.mean():.4f} "
        f"within0.10 xgb={float(np.mean(ex < 0.10)):.3f}"
    )
    brier = lambda a, b: float(np.mean((a - b) ** 2))
    print(
        f"q_m: Brier xgb={brier(q_hat, q_real):.4f} naive={brier(naive_q, q_real):.4f} "
        f"agree xgb={float(np.mean((q_hat > 0.5) == (q_real > 0.5))):.3f} "
        f"naive={float(np.mean((naive_q > 0.5) == (q_real > 0.5))):.3f}"
    )
    buckets: dict[str, dict[str, float]] = {}
    for lo, hi in ((3, 7), (8, 12), (13, 17), (18, 20)):
        m = (ks[te] >= lo) & (ks[te] <= hi)
        buckets[f"{lo}-{hi}"] = {
            "mae_s": float(np.abs(s_hat[m] - s_true[m]).mean()),
            "mae_q": float(np.abs(q_hat[m] - q_real[m]).mean()),
        }
        print(
            f"k={lo:2d}..{hi:2d} n={int(m.sum()):4d} "
            f"MAE_S xgb={buckets[f'{lo}-{hi}']['mae_s']:.4f} "
            f"MAE_q xgb={buckets[f'{lo}-{hi}']['mae_q']:.4f}"
        )

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    metrics = {
        "mae_s": float(ex.mean()),
        "brier_q": brier(q_hat, q_real),
        "buckets": buckets,
        "features": FEATURES,
        "targets": TARGETS,
    }
    METRICS_OUT = MODEL_OUT.with_name("xgb_partial_metrics.json")
    METRICS_OUT.write_text(json.dumps(metrics, indent=2))
    print(f"saved={MODEL_OUT.resolve()}")
    print(f"saved={METRICS_OUT.resolve()}")

    for mo in months:
        n = len(mo["c"])
        if n >= MIN_N or n < K_MIN:
            continue
        t_hat = model.predict(
            np.array(
                build_partial_features(mo["h"], mo["low"], mo["c"], mo["c0"], n)
            ).reshape(1, -1)
        )[0]
        print(
            f"nowcast {mo['ym']} (k={n}): S_m~{geomean_row(t_hat):.3f} "
            f"q_m~{float(gmm_full.predict_proba(np.clip(t_hat, 0, 1).reshape(1, -1))[0, tidx_full]):.3f}"
        )


if __name__ == "__main__":
    main()
