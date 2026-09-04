# XAUUSD Next-Month Regime Forecast

## 1. Objective and forecast origin

The system predicts whether the **next complete calendar month** will exhibit a
persistent trend or sideways/choppy price behavior. It does not predict bullish
or bearish direction and it does not issue a trading signal.

After the final XAUUSD D1 bar of month $m$, freeze one forecast for month
$m+1$:

\[
p_{m+1}=P(Z_{m+1}=\text{Trend}\mid\mathcal F_m),
\qquad
P(Z_{m+1}=\text{Sideways})=1-p_{m+1}.
\]

There are two distinct stages:

1. measure the realized, model-defined regime of a completed month;
2. predict the following month's realized regime using only information
   available at the forecast origin.

## 2. Data policy

- Use raw XAUUSD D1 OHLC from one consistent feed.
- Use real bars only (`is_filled=False`). Filled holiday placeholders are not
  observations.
- A month is valid for regime fitting and calendar estimates only when it is
  complete and contains at least 15 real bars.
- Use UTC dates. The first available prior close may fall back to the first
  bar's Open; that partial first month remains outside all fitted samples.
- Heikin-Ashi and other smoothed prices are excluded.

## 3. Realized monthly regime features

For a valid month with $N$ daily bars, let $C_0$ be the final real Close
before the month and let $H_t,L_t,C_t$ denote its daily prices.

### Range efficiency

\[
TR_t=\max(H_t,C_{t-1})-\min(L_t,C_{t-1}),
\]

\[
R_m=\max(C_0,H_1,\ldots,H_N)-\min(C_0,L_1,\ldots,L_N),
\]

\[
T_{\mathrm{range}}
=1-\frac{\ln\left(\sum_{t=1}^{N}TR_t/R_m\right)}{\ln N}.
\]

High values indicate that cumulative travel efficiently expanded the monthly
envelope. Before clipping, validate $R_m>0$, $\sum TR_t>0$, and allow only
floating-point-sized bound violations. Material violations are data or formula
errors.

### Directional coherence

With close-to-close log returns

\[
r_t=\ln(C_t/C_{t-1}),
\]

define

\[
T_{\mathrm{direction}}
=\frac{\left|\sum_{t=1}^{N}r_t\right|}
{\sqrt{N\sum_{t=1}^{N}r_t^2}}.
\]

This combines directional agreement and movement participation:

\[
T_{\mathrm{direction}}
=\frac{|\sum r_t|}{\sum|r_t|}
\times
\frac{\sum|r_t|}{\sqrt{N\sum r_t^2}}.
\]

A single nonzero return scores $1/\sqrt N$, regardless of its magnitude.

### Temporal monotonicity

\[
T_{\mathrm{mono}}
=\left|\tau_b([0,1,\ldots,N],[C_0,C_1,\ldots,C_N])\right|.
\]

Kendall tau is rank-based, so using log prices would give the same result.
This feature penalizes trend-then-reversal paths that return distributions alone
cannot distinguish.

The three features are complementary, not statistically independent:

\[
X_m=[T_{\mathrm{range}},T_{\mathrm{direction}},T_{\mathrm{mono}}].
\]

For diagnostics only, use

\[
S_m=(T_{\mathrm{range}}T_{\mathrm{direction}}T_{\mathrm{mono}})^{1/3}.
\]

For $N\le1$, return zeros. A completely flat month is invalid. Valid
normalized values are clipped to $[0,1]$ only after invariant checks.

### Forecast-origin price state

From the same real D1 returns, retain price-only information known at the end
of the origin month:

\[
RV_m=\sum_t r_t^2,
\qquad
PK_m=\frac{1}{4\ln2}\sum_t\ln^2(H_t/L_t),
\]

\[
J_m=\frac{\max_t r_t^2}{RV_m},
\qquad
A_m=\frac{|\sum_{r_t\ge0}r_t^2-\sum_{r_t<0}r_t^2|}{RV_m}.
\]

The multiscale predictors use log $RV$ over the last 1, 3, and 12 complete
months, log current-month $PK$, current $J_m$ and $A_m$, $q_m$, and
$q_m-q_{m-1}$. Rolling windows must contain consecutive valid months and must
end at the forecast origin.

## 4. Prior-only realized state

The operational outcome is a **model-defined latent regime**, not independently
observed ground truth. For each valid month $t$:

1. require at least 48 earlier valid months;
2. fit a two-component full-covariance Gaussian mixture to $X_{1:t-1}$, with
   `n_init=10`, `reg_covar=1e-3`, and `random_state=0`;
3. identify Trend as the component with the larger mean across the three
   equally scaled features;
4. record
   \[
   q_t=P(Z_t=\text{Trend}\mid X_t, X_{1:t-1}).
   \]

No later month may revise $q_t$. Report whether the Trend component dominates
the other component in every feature and whether two components beat one by
BIC. Crossed component means or fewer than 80% BIC-supporting evaluation folds
fail the regime-stability gate.

The full-history GMM and calendar table remain useful descriptive views, but
they must never supply outcomes for a forecasting-skill claim.

## 5. Forecast candidates

Each outer evaluation origin uses expanding history and predicts exactly one
month ahead. Training pairs must be consecutive valid calendar months.
Evaluation begins after at least 36 prior transition examples.

Compare:

1. a Jeffreys-smoothed global prior;
2. a Jeffreys-smoothed target-calendar-month prior;
3. a prior-only fractional-logistic calibration of the calendar prior;
4. persistence, $p_{t+1}=q_t$;
5. a globally shrunk two-state soft Markov transition;
6. target-month soft Markov transitions shrunk toward the global transition;
7. an L2-regularized fractional-logistic model using the calendar prior,
   $q_t$, and $q_t-q_{t-1}$;
8. first- and second-harmonic cyclic seasonality;
9. twelve centered calendar-month effects with L2 partial pooling;
10. the multiscale price state without seasonality;
11. cyclic seasonality plus the multiscale price state;
12. partially pooled month effects plus the multiscale price state.

For soft Markov transitions, expected counts use the fractional state weights.
For example,

\[
\hat a_1
=\frac{0.5+\sum_t q_tq_{t+1}}{1+\sum_t q_t},
\qquad
\hat a_0
=\frac{0.5+\sum_t(1-q_t)q_{t+1}}{1+\sum_t(1-q_t)},
\]

and

\[
p_{t+1}=(1-q_t)\hat a_0+q_t\hat a_1.
\]

Choose seasonal shrinkage from \(\{2,6,12,24\}\) and logistic L2 penalty from
\(\{0.01,0.1,1,10,100\}\) using inner expanding-window Brier loss only.

## 6. Evaluation and model promotion

Use monthly expanding-window evaluation. Random cross-validation is forbidden.
The primary score for forecast $p_t$ and soft outcome $q_t$ is expected
binary Brier loss:

\[
L_B=(p_t-q_t)^2+q_t(1-q_t).
\]

Also report fractional log loss, hard agreement at 0.5, reliability bins, and
calibration diagnostics. Compare each candidate with the calendar baseline via
a paired year-cluster bootstrap with a fixed seed.

A 95% block-bootstrap Model Confidence Set sequentially removes demonstrably
inferior models using their common-fold Brier losses. A candidate becomes the
official model only when:

- the regime-stability gate passes; and
- calendar is excluded from the final Model Confidence Set.

Among surviving candidates, select the lowest Brier loss and prefer the simpler
model in a tie. If calendar remains in the confidence set, the official output
is the calendar baseline and must say `baseline_fallback`. A sophisticated
model is not promoted merely because its point estimate is best.

## 7. Live output and uncertainty

The live artifact contains:

- forecast origin and its final D1 date;
- target calendar month;
- official model and calendar baseline;
- $P(\text{Trend})$ and its exact complementary
  $P(\text{Sideways})$;
- a deterministic 95% year-cluster bootstrap interval;
- training sample size, target definition, and diagnostic status.

The candidate artifact additionally reports every model's forecast, interval
width, walk-forward scores, confidence-set membership, and eligibility.

The interval represents sampling/model uncertainty conditional on the chosen
latent-regime definition. It is not an interval for trading returns.

## 8. Required validation

The implementation must preserve scale and time-reversal invariance, bull/bear
symmetry, the single-shock penalty, random-walk behavior, and the `N>=15` gate.
It must additionally prove that future observations cannot alter earlier
labels or forecasts, every transition is between consecutive months, all
probabilities are finite and bounded, outputs are deterministic, and model
selection falls back safely when no challenger credibly beats calendar.
