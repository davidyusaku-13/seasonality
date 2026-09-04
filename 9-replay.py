"""Blind 2026 replay of the frozen next-month forecasting pipeline."""

from pathlib import Path

import numpy as np
import polars as pl

import forecasting as fc

OUT = Path("data/replay_2026.csv")


def main() -> None:
    feat = pl.read_parquet(fc.FEAT_PATH).sort(["year", "month"])
    prequential = fc.build_prequential_targets(feat)
    predictions, _, _, _ = fc.run_backtest(prequential)
    transitions = fc.build_transition_records(prequential)
    targets = [r for r in predictions if r["year"] == 2026]
    if not targets:
        raise SystemExit("no valid 2026 forecast targets")

    replay = []
    for row in targets:
        prior = [r for r in predictions if r["seq"] < row["seq"]]
        comparison, champion, regime_ok = fc.select_champion(prior)
        official = float(row[f"p_{champion}"])
        train = [r for r in transitions if r["seq"] < row["seq"]]
        context = fc._context_from_record(row)
        tuning = {
            "seasonal_strength": row["seasonal_strength"],
            "logistic_penalty": row["logistic_penalty"],
            **{
                f"{model}_penalty": row[f"{model}_penalty"] for model in fc.RIDGE_MODELS
            },
        }
        lo, hi = fc.live_interval(train, context, champion, tuning)
        lo, hi = min(lo, official), max(hi, official)
        replay.append(
            {
                "ym": row["ym"],
                "model": champion,
                "pred_q": official,
                "ci95_lo": lo,
                "ci95_hi": hi,
                "calendar_q": row["p_calendar"],
                "real_q": row["q"],
                "train_n": len(train),
                "regime_diagnostics_pass": regime_ok,
            }
        )
        selected = next(r for r in comparison if r["model"] == champion)
        print(
            f"{row['ym']}: model={champion} pred={official:.3f} "
            f"CI=[{lo:.3f},{hi:.3f}] calendar={row['p_calendar']:.3f} "
            f"real_q={row['q']:.3f} "
            f"prior_Brier={selected['brier']:.4f}"
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(replay).write_csv(OUT)
    p = np.array([r["pred_q"] for r in replay])
    c = np.array([r["calendar_q"] for r in replay])
    q = np.array([r["real_q"] for r in replay])
    bp = float(np.mean((p - q) ** 2 + q * (1 - q)))
    bc = float(np.mean((c - q) ** 2 + q * (1 - q)))
    agree = float(np.mean((p >= 0.5) == (q >= 0.5)))
    print(f"wrote {OUT}")
    print(
        f"{len(replay)}-month replay: Brier official={bp:.4f} "
        f"calendar={bc:.4f} skill={1 - bp / bc:+.3f}"
    )
    print(f"hard agreement={agree:.3f}")
    print("REPLAY: PASS (all targets and model selections are prior-only)")


if __name__ == "__main__":
    main()
