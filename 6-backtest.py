"""Walk-forward verification of the calendar-month projection (no lookahead).

For each test year Y in 2012..2026: fit the 2-component GMM and the calendar
means on months strictly before Jan Y (N>=15), predict P(Trend) for each
month of Y, and score against the realized posterior from that same
pre-Y model. Compares soft Brier score vs a climatology baseline (train
mean) plus hard accuracy and calibration bins.

Exit 0 always; this measures skill, it does not gate.
"""

from pathlib import Path

import numpy as np
import polars as pl
from sklearn.mixture import GaussianMixture

FEAT = Path("data/monthly_features.parquet")
MIN_N = 15
FIRST_TEST_YEAR = 2012
FEATS = ["t_range", "t_direction", "t_mono"]


def fit_gmm(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gmm = GaussianMixture(
        n_components=2,
        covariance_type="full",
        n_init=10,
        random_state=0,
        reg_covar=1e-3,
    )
    gmm.fit(X)
    trend_idx = int(np.argsort(gmm.means_.mean(axis=1))[1])
    return gmm, gmm.predict_proba(X)[:, trend_idx]


def main() -> None:
    feat = pl.read_parquet(FEAT).sort(["year", "month"])
    years = sorted(set(feat["year"].to_list()))
    recs = []
    for y in [t for t in years if t >= FIRST_TEST_YEAR]:
        train = feat.filter((pl.col("year") < y) & (pl.col("n") >= MIN_N))
        if train.height < 24:
            continue
        Xtr = train.select(FEATS).to_numpy()
        gmm, qtr = fit_gmm(Xtr)
        cal = {}
        for m in range(1, 13):
            qq = qtr[train["month"].to_numpy() == m]
            cal[m] = float((0.5 + qq.sum()) / (len(qq) + 1)) if len(qq) else 0.5
        clima = float(qtr.mean())
        test = feat.filter((pl.col("year") == y) & (pl.col("n") >= MIN_N))
        Xte = test.select(FEATS).to_numpy()
        trend_idx = int(np.argsort(gmm.means_.mean(axis=1))[1])
        qre = gmm.predict_proba(Xte)[:, trend_idx]
        for row, qr in zip(test.to_dicts(), qre):
            recs.append((y, row["month"], cal[row["month"]], float(qr), clima))

    p = np.array([r[2] for r in recs])
    r_ = np.array([r[3] for r in recs])
    c = np.array([r[4] for r in recs])
    brier, brier_clima = float(np.mean((p - r_) ** 2)), float(np.mean((c - r_) ** 2))
    acc = float(np.mean((p > 0.5) == (r_ > 0.5)))
    acc_clima = float(np.mean((c > 0.5) == (r_ > 0.5)))
    print(f"predictions: {len(recs)} month-years")
    print(
        f"Brier calendar={brier:.4f} climatology={brier_clima:.4f} "
        f"skill={1 - brier / brier_clima:+.3f}"
    )
    print(f"hard accuracy calendar={acc:.3f} climatology={acc_clima:.3f}")
    print("calibration (bin: n, mean_pred, mean_realized):")
    for lo, hi in ((0.0, 0.33), (0.33, 0.66), (0.66, 1.01)):
        m = (p >= lo) & (p < hi)
        if m.sum():
            print(
                f"  [{lo:.2f},{hi:.2f}): n={int(m.sum())} "
                f"pred={p[m].mean():.3f} realized={r_[m].mean():.3f}"
            )


if __name__ == "__main__":
    main()
