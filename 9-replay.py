"""Blind replay Jan 2026 -> latest complete month (no lookahead).

For each target month M: fit GMM + calendar means + XGB partial model on
months strictly before M (N>=15), project M from its calendar row, nowcast
M at every k from the step's XGB model, then confirm against M's realized
q (step-GMM posterior) and S (deterministic formula on revealed data).

Output: data/replay_2026.csv (kind in {projection, nowcast})
"""

import importlib.util
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.mixture import GaussianMixture
from sklearn.multioutput import MultiOutputRegressor


def load_mod(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s2 = load_mod("2-seasonality.py")
xgb8 = load_mod("8-xgb.py")

SRC = Path("data/xauusd_d1.parquet")
OUT = Path("data/replay_2026.csv")
MIN_N = 15
TARGETS = ["t_range", "t_direction", "t_mono"]


def final_X(mo: dict) -> np.ndarray:
    tr, td, tm, _ = s2.month_features(mo["h"], mo["low"], mo["c"], mo["c0"])
    return np.array([tr, td, tm])


def fit_gmm(X: np.ndarray):
    g = GaussianMixture(
        n_components=2,
        covariance_type="full",
        n_init=10,
        random_state=0,
        reg_covar=1e-3,
    )
    g.fit(X)
    tidx = int(np.argsort(g.means_.mean(axis=1))[1])
    return g, tidx, g.predict_proba(X)[:, tidx]


def main() -> None:
    real = pl.read_parquet(SRC).filter(~pl.col("is_filled")).sort("time")
    months = [m for m in xgb8.month_rows(real) if len(m["c"]) >= MIN_N]
    by_ym = {m["ym"]: m for m in months}
    targets = [f"2026-{m:02d}" for m in range(1, 9)]
    assert all(t in by_ym for t in targets), "missing 2026 months"

    recs = []
    for t in targets:
        hist = [m for m in months if m["ym"] < t]
        Xh = np.array([final_X(m) for m in hist])
        g, tidx, qh = fit_gmm(Xh)
        qh_by_m: dict[int, list] = {}
        for m, q in zip([m["month"] for m in hist], qh):
            qh_by_m.setdefault(m, []).append(q)
        mo = by_ym[t]
        p = float((0.5 + sum(qh_by_m[mo["month"]])) / (len(qh_by_m[mo["month"]]) + 1))
        clima = float(qh.mean())
        X_t = final_X(mo)
        q_real = float(g.predict_proba(X_t.reshape(1, -1))[0, tidx])
        s_real = xgb8.geomean_row(X_t)

        Xw, Yw = [], []
        for h in hist:
            n = len(h["c"])
            for k in range(3, min(20, n - 1) + 1):
                Xw.append(
                    xgb8.build_partial_features(h["h"], h["low"], h["c"], h["c0"], k)
                )
                Yw.append(final_X(h))
        Xw, Yw = np.array(Xw), np.array(Yw)
        xgb_model = MultiOutputRegressor(
            xgb.XGBRegressor(
                objective="reg:squarederror",
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                seed=0,
                n_jobs=1,
            )
        )
        xgb_model.fit(Xw, Yw)
        recs.append(
            {
                "ym": t,
                "kind": "projection",
                "k": 0,
                "pred_q": p,
                "pred_s": np.nan,
                "real_q": q_real,
                "real_s": s_real,
                "clima": clima,
            }
        )
        nows = {}
        for k in range(3, len(mo["c"])):
            f = np.array(
                xgb8.build_partial_features(mo["h"], mo["low"], mo["c"], mo["c0"], k)
            ).reshape(1, -1)
            th = xgb_model.predict(f)[0]
            nows[k] = (
                float(g.predict_proba(np.clip(th, 0, 1).reshape(1, -1))[0, tidx]),
                xgb8.geomean_row(th),
            )
            recs.append(
                {
                    "ym": t,
                    "kind": "nowcast",
                    "k": k,
                    "pred_q": nows[k][0],
                    "pred_s": nows[k][1],
                    "real_q": q_real,
                    "real_s": s_real,
                    "clima": clima,
                }
            )
        show = " ".join(f"@k{k}={nows[k][0]:.3f}" for k in (5, 10, 15, 20) if k in nows)
        print(
            f"{t}: proj={p:.3f} clima={clima:.3f} real_q={q_real:.3f} "
            f"real_S={s_real:.3f} now_q {show}"
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(recs).write_csv(OUT)
    proj = [r for r in recs if r["kind"] == "projection"]
    bp = np.mean([(r["pred_q"] - r["real_q"]) ** 2 for r in proj])
    bc = np.mean([(r["clima"] - r["real_q"]) ** 2 for r in proj])
    print(f"wrote {OUT.resolve()}")
    print(
        f"8-month replay: Brier proj={bp:.4f} clima={bc:.4f} skill={1 - bp / bc:+.3f}"
    )
    print(
        f"hard agree proj={np.mean([(r['pred_q'] > 0.5) == (r['real_q'] > 0.5) for r in proj]):.3f}"
    )


if __name__ == "__main__":
    main()
