"""Blind replay Jan 2026 -> latest complete month (no lookahead).

For each target month M: fit GMM + calendar means on months strictly
before M (N>=15), project M from its calendar row, then confirm against
M's realized q (step-GMM posterior) and S (formula on revealed data).

Output: data/replay_2026.csv (one row per target month)
"""

import importlib.util
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.mixture import GaussianMixture


def load_mod(name: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s2 = load_mod("2-seasonality.py")

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
    months = [m for m in s2.month_rows(real) if len(m["c"]) >= MIN_N]
    by_ym = {m["ym"]: m for m in months}
    targets = [f"2026-{m:02d}" for m in range(1, 9)]
    assert all(t in by_ym for t in targets), "missing 2026 months"

    recs = []
    stab = []
    for t in targets:
        hist = [m for m in months if m["ym"] < t]
        Xh = np.array([final_X(m) for m in hist])
        g, tidx, qh = fit_gmm(Xh)
        stab.append(
            {
                "ym": t,
                "means": g.means_.copy(),
                "weights": g.weights_.copy(),
                "tidx": tidx,
            }
        )
        qh_by_m: dict[int, list] = {}
        for m, q in zip([m["month"] for m in hist], qh):
            qh_by_m.setdefault(m, []).append(q)
        mo = by_ym[t]
        p = float((0.5 + sum(qh_by_m[mo["month"]])) / (len(qh_by_m[mo["month"]]) + 1))
        clima = float(qh.mean())
        X_t = final_X(mo)
        q_real = float(g.predict_proba(X_t.reshape(1, -1))[0, tidx])
        s_real = s2.geomean_row(X_t)
        recs.append(
            {
                "ym": t,
                "pred_q": p,
                "real_q": q_real,
                "real_s": s_real,
                "clima": clima,
            }
        )
        print(
            f"{t}: proj={p:.3f} clima={clima:.3f} real_q={q_real:.3f} "
            f"real_S={s_real:.3f}"
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(recs).write_csv(OUT)
    bp = np.mean([(r["pred_q"] - r["real_q"]) ** 2 for r in recs])
    bc = np.mean([(r["clima"] - r["real_q"]) ** 2 for r in recs])
    print(f"wrote {OUT.resolve()}")
    print(
        f"8-month replay: Brier proj={bp:.4f} clima={bc:.4f} skill={1 - bp / bc:+.3f}"
    )
    print(
        f"hard agree proj={np.mean([(r['pred_q'] > 0.5) == (r['real_q'] > 0.5) for r in recs]):.3f}"
    )

    ref_g, ref_tidx, _ = fit_gmm(np.array([final_X(m) for m in months]))
    ref_means = ref_g.means_
    print("refit stability (drift = L2 of trend-mean vs full-data fit):")
    worst, ok = 0.0, True
    for s in stab:
        D = np.array(
            [
                [np.linalg.norm(s["means"][i] - ref_means[j]) for j in (0, 1)]
                for i in (0, 1)
            ]
        )
        match_trend = int(np.argmin(D[:, ref_tidx]))  # step comp nearest ref-trend
        drift = float(np.linalg.norm(s["means"][s["tidx"]] - ref_means[ref_tidx]))
        worst = max(worst, drift)
        same = match_trend == s["tidx"] and float(s["weights"].min()) > 0.2
        ok &= same
        print(
            f"  {s['ym']}: tidx={s['tidx']} weights={np.round(s['weights'], 3).tolist()} "
            f"drift={drift:.4f} label_match={match_trend == s['tidx']}"
        )
    print(
        f"STABILITY: {'PASS' if ok and worst < 0.05 else 'FAIL'} (max_drift={worst:.4f})"
    )


if __name__ == "__main__":
    main()
