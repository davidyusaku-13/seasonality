"""Partial-month scorer: first-k-days data -> final full-month S_m.

Trains XGBoost on complete months (N>=15): for each month and each
k in 3..min(20, N-1), features come strictly from the first k days plus
the prior close C0 (known at month start), target is the final S_m.
Time split (train years < 2021, test >= 2021), naive baseline = partial S.
Saves models/xgb_partial.json and nowcasts any in-progress month.

Outputs: models/xgb_partial.json
"""

import importlib.util
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb

SPEC = Path(__file__).with_name("2-seasonality.py")
_spec = importlib.util.spec_from_file_location("seasonality2", SPEC)
s2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s2)

SRC = Path("data/xauusd_d1.parquet")
FEAT = Path("data/monthly_features.parquet")
MODEL_OUT = Path("models/xauusd_partial.json")
MIN_N = 15
K_MIN, K_MAX = 3, 20
SPLIT_YEAR = 2021

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


def build_dataset(
    months: list[dict], s_by_ym: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    X, y, ks, keys = [], [], [], []
    for mo in months:
        n = len(mo["c"])
        if n < MIN_N or mo["ym"] not in s_by_ym:
            continue
        for k in range(K_MIN, min(K_MAX, n - 1) + 1):
            X.append(build_partial_features(mo["h"], mo["low"], mo["c"], mo["c0"], k))
            y.append(s_by_ym[mo["ym"]])
            ks.append(k)
            keys.append((mo["ym"], k))
    return np.array(X), np.array(y), np.array(ks), keys


def report(name: str, pred: np.ndarray, actual: np.ndarray) -> None:
    err = np.abs(pred - actual)
    ss_res = float(np.sum((actual - pred) ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    print(
        f"{name}: MAE={err.mean():.4f} RMSE={np.sqrt((err**2).mean()):.4f} "
        f"R2={1 - ss_res / ss_tot:.3f} within0.10={float(np.mean(err < 0.10)):.3f}"
    )


def main() -> None:
    real = pl.read_parquet(SRC).filter(~pl.col("is_filled")).sort("time")
    s_by_ym = {r["ym"]: float(r["s"]) for r in pl.read_parquet(FEAT).to_dicts()}
    months = month_rows(real)
    X, y, ks, _ = build_dataset(months, s_by_ym)
    ym_of_row = []
    for mo in months:
        n = len(mo["c"])
        if n < MIN_N or mo["ym"] not in s_by_ym:
            continue
        for _k in range(K_MIN, min(K_MAX, n - 1) + 1):
            ym_of_row.append(mo["ym"])
    ym_of_row = np.array(ym_of_row)
    tr = ym_of_row < f"{SPLIT_YEAR}-01"
    te = ~tr
    print(f"rows={len(y)} train={int(tr.sum())} test={int(te.sum())}")

    model = xgb.XGBRegressor(
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
    model.fit(X[tr], y[tr])
    pred = model.predict(X[te])
    naive = X[te][:, FEATURES.index("pS")]
    print("--- test (>= 2021) ---")
    report("xgb  ", pred, y[te])
    report("naive", naive, y[te])
    for lo, hi in ((3, 7), (8, 12), (13, 17), (18, 20)):
        m = (ks[te] >= lo) & (ks[te] <= hi)
        err_x = np.abs(pred[m] - y[te][m]).mean()
        err_n = np.abs(naive[m] - y[te][m]).mean()
        print(
            f"k={lo:2d}..{hi:2d} n={int(m.sum()):4d} MAE xgb={err_x:.4f} naive={err_n:.4f}"
        )
    imp = sorted(zip(FEATURES, model.feature_importances_), key=lambda t: -t[1])
    print("importance:", [(f, round(float(v), 3)) for f, v in imp])

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_OUT)
    print(f"saved={MODEL_OUT.resolve()}")

    import json

    buckets: dict[str, float] = {}
    for lo, hi in ((3, 7), (8, 12), (13, 17), (18, 20)):
        m = (ks[te] >= lo) & (ks[te] <= hi)
        buckets[f"{lo}-{hi}"] = float(np.abs(pred[m] - y[te][m]).mean())
    metrics_path = MODEL_OUT.with_name("xgb_partial_metrics.json")
    metrics_path.write_text(
        json.dumps(
            {
                "test_mae": float(np.abs(pred - y[te]).mean()),
                "mae_by_k_bucket": buckets,
                "features": FEATURES,
            },
            indent=2,
        )
    )
    print(f"saved={metrics_path.resolve()}")

    for mo in months:
        n = len(mo["c"])
        if n >= MIN_N or n < K_MIN:
            continue
        f = np.array(
            build_partial_features(mo["h"], mo["low"], mo["c"], mo["c0"], n)
        ).reshape(1, -1)
        print(
            f"nowcast {mo['ym']} (k={n}): predicted final S_m={float(model.predict(f)[0]):.3f}"
        )


if __name__ == "__main__":
    main()
