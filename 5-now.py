"""Frozen month-end forecast for the next XAUUSD monthly regime."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import polars as pl

import forecasting as fc

OUT = Path("charts/now.png")
FORECAST_OUT = Path("data/next_month_forecast.csv")
CANDIDATES_OUT = Path("data/next_month_candidates.csv")
TRAIL = 12


def main() -> None:
    matplotlib.rcParams["axes.grid"] = False
    feat = pl.read_parquet(fc.FEAT_PATH).sort(["year", "month"])
    prequential = fc.build_prequential_targets(feat)
    backtest, comparison, champion, regime_ok = fc.run_backtest(prequential)
    rows, context, latest = fc.live_context(prequential)
    predictions, tuning = fc.predict_candidates(rows, context)
    comparison_by_model = {r["model"]: r for r in comparison}
    calendar_in_mcs = comparison_by_model["calendar"]["mcs_included"]
    candidate_rows = []
    for model in fc.CANDIDATES:
        candidate_p = predictions[model]
        candidate_lo, candidate_hi = fc.live_interval(rows, context, model, tuning)
        candidate_lo = min(candidate_lo, candidate_p)
        candidate_hi = max(candidate_hi, candidate_p)
        metrics = comparison_by_model[model]
        candidate_rows.append(
            {
                "forecast_origin": latest["ym"],
                "target_month": f"{context.target_year}-{context.target_month:02d}",
                "model": model,
                "p_trend": candidate_p,
                "ci95_lo": candidate_lo,
                "ci95_hi": candidate_hi,
                "interval_width": candidate_hi - candidate_lo,
                "oos_brier": metrics["brier"],
                "oos_log_loss": metrics["log_loss"],
                "calibration_ece_3bin": metrics["calibration_ece_3bin"],
                "mcs_included": metrics["mcs_included"],
                "promotion_eligible": bool(
                    regime_ok
                    and not calendar_in_mcs
                    and metrics["mcs_included"]
                    and model in fc.PROMOTABLE
                ),
                "selected": model == champion,
                "train_n": len(rows),
            }
        )
    p = predictions[champion]
    selected_candidate = next(r for r in candidate_rows if r["selected"])
    lo, hi = selected_candidate["ci95_lo"], selected_candidate["ci95_hi"]

    target = f"{context.target_year}-{context.target_month:02d}"
    origin = latest["ym"]
    status = "validated" if champion != "calendar" else "baseline_fallback"
    result = {
        "forecast_origin": origin,
        "origin_end": latest["end"],
        "target_month": target,
        "p_trend": p,
        "p_sideways": 1 - p,
        "ci95_lo": lo,
        "ci95_hi": hi,
        "model": champion,
        "baseline_p": predictions["calendar"],
        "train_n": len(rows),
        "target_definition": "prior-only GMM latent trend state",
        "diagnostic_status": status if regime_ok else "regime_diagnostics_failed",
    }
    FORECAST_OUT.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame([result]).write_csv(FORECAST_OUT)
    pl.DataFrame(candidate_rows).write_csv(CANDIDATES_OUT)

    print(f"forecast origin: {origin} final D1 bar {latest['end']}")
    print(f"target month: {target}")
    print(
        f"P(Trend)={p:.1%} P(Sideways)={1 - p:.1%} "
        f"95% bootstrap interval=[{lo:.1%},{hi:.1%}]"
    )
    print(
        f"model={champion} calendar_baseline={predictions['calendar']:.1%} "
        f"status={result['diagnostic_status']} train_n={len(rows)}"
    )
    print(f"walk-forward forecasts used for selection: {len(backtest)}")

    scored = [r for r in prequential if r["q_prequential"] is not None][-TRAIL:]
    xs = list(range(len(scored) + 1))
    labels = [r["ym"] for r in scored] + [f"{target}*"]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        xs[:-1],
        [r["q_prequential"] for r in scored],
        marker="o",
        linewidth=2,
        label="realized prior-only q",
    )
    ax.errorbar(
        [xs[-1]],
        [p],
        yerr=[[p - lo], [hi - p]],
        fmt="s",
        color="green",
        ecolor="black",
        capsize=5,
        label=f"next-month forecast ({champion})",
    )
    if champion != "calendar":
        ax.scatter(
            [xs[-1]],
            [predictions["calendar"]],
            marker="x",
            color="gray",
            label="calendar baseline",
        )
    for boundary in [x - 0.5 for x in xs] + [xs[-1] + 0.5]:
        ax.axvline(boundary, color="gray", linestyle="--", linewidth=0.8, alpha=0.35)
    ax.set_ylim(0, 1)
    ax.set_xlim(xs[0] - 0.5, xs[-1] + 0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("0=sideways/choppy, 1=trend")
    ax.set_title(
        f"XAUUSD next-month regime forecast: {target} (origin {origin}, {champion})"
    )
    ax.legend()
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    plt.close(fig)
    print(f"wrote {FORECAST_OUT}, {CANDIDATES_OUT}, and {OUT}")

    selected = next(r for r in comparison if r["model"] == champion)
    print(
        f"selected walk-forward Brier={selected['brier']:.4f} "
        f"logloss={selected['log_loss']:.4f}"
    )


if __name__ == "__main__":
    main()
