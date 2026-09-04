"""Monthly expanding-window evaluation of next-month regime forecasts."""

from pathlib import Path

import polars as pl

import forecasting as fc

PRED_OUT = Path("data/forecast_backtest.csv")
MODEL_OUT = Path("data/model_comparison.csv")


def main() -> None:
    feat = pl.read_parquet(fc.FEAT_PATH).sort(["year", "month"])
    prequential = fc.build_prequential_targets(feat)
    predictions, comparison, champion, regime_ok = fc.run_backtest(prequential)
    if not predictions:
        raise SystemExit("insufficient history for forecast backtest")

    PRED_OUT.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(predictions).write_csv(PRED_OUT)
    table = pl.DataFrame(comparison).with_columns(
        (pl.col("model") == champion).alias("selected"),
        pl.lit(regime_ok).alias("regime_diagnostics_pass"),
    )
    table.write_csv(MODEL_OUT)

    crossed = sum(r["component_crossed"] for r in predictions)
    bic = sum(r["bic_advantage_2"] > 0 for r in predictions)
    print(f"predictions: {len(predictions)} monthly forecast origins")
    print(
        f"regime diagnostics: crossed={crossed}/{len(predictions)} "
        f"BIC2_support={bic}/{len(predictions)} "
        f"status={'PASS' if regime_ok else 'FAIL'}"
    )
    print(
        f"{'model':<24} {'Brier':>8} {'logloss':>8} {'agree':>8} {'MCS':>5} "
        f"{'dBrier vs cal [95% CI]':>30}"
    )
    for row in comparison:
        marker = " *" if row["model"] == champion else ""
        print(
            f"{row['model']:<24} {row['brier']:>8.4f} "
            f"{row['log_loss']:>8.4f} {row['hard_agreement']:>8.3f} "
            f"{row['mcs_included']!s:>5} "
            f"{row['brier_diff_vs_calendar']:>+8.4f} "
            f"[{row['brier_diff_ci95_lo']:+.4f},{row['brier_diff_ci95_hi']:+.4f}]"
            f"{marker}"
        )
    print("calibration (intercept, slope, 3-bin ECE):")
    for row in comparison:
        print(
            f"  {row['model']:<16} {row['calibration_intercept']:+.3f} "
            f"{row['calibration_slope']:.3f} {row['calibration_ece_3bin']:.3f}"
        )
    reason = (
        "calendar excluded from the 95% model confidence set"
        if champion != "calendar"
        else "calendar remains in the 95% model confidence set"
    )
    print(f"CHAMPION: {champion} ({reason})")
    print(f"wrote {PRED_OUT} and {MODEL_OUT}")


if __name__ == "__main__":
    main()
