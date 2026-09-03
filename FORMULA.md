# Forex Monthly Regime Seasonality Formula

## Objective

The goal is to estimate, for each calendar month, the historical probability that the market is:

- **Trending**, or
- **Sideways / choppy**

This is **not** directional seasonality.

We are not asking questions such as:

> Does XAUUSD usually go up in January?

Instead, the question is:

> How often does XAUUSD behave like a persistent trend during January, regardless of whether the trend is bullish or bearish?

The final output should look conceptually like:

| Month | P(Trend) | P(Sideways) |
|---|---:|---:|
| January | 68% | 32% |
| February | 44% | 56% |
| March | 72% | 28% |
| ... | ... | ... |

The classification should be derived entirely from **price behavior**, not from strategy performance.

---

# 1. Timeframe

Use:

\[
\boxed{\text{D1}}
\]

Each calendar month therefore contains approximately 20–23 observations.

D1 is preferred because the research question concerns the regime of the **entire month**, not intraday market structure.

Using lower timeframes such as M15 would heavily overweight intraday noise and session structure.

---

# 2. Price Representation

## 2.1 Line chart

A conventional line chart normally plots one price per bar.

The default is usually:

\[
\boxed{\text{Close}}
\]

Therefore, the daily Close sequence is a natural representation of the month's directional trajectory.

## 2.2 Japanese candlesticks

Japanese candlesticks contain:

\[
O_t,\ H_t,\ L_t,\ C_t
\]

They preserve information that a line chart cannot show, especially:

- intraday excursions,
- overlapping ranges,
- large wicks,
- price travel that eventually returns near the Close.

This information is useful for determining whether a month is genuinely clean or highly choppy.

## 2.3 Heikin-Ashi

Heikin-Ashi should **not** be used for this research.

Heikin-Ashi transforms and smooths raw OHLC data.

That is useful visually, but inappropriate for a regime classifier because smoothing makes trends appear cleaner than the original market actually was.

The classifier should measure the raw market rather than first transforming it into something trend-like.

---

# 3. Which OHLC Fields Are Needed?

Use raw D1 OHLC data as the source dataset.

However, the core formula does not require every OHLC field equally.

The important values are:

\[
\boxed{H_t,\ L_t,\ C_t,\ C_{t-1}}
\]

Open is not currently required as a separate feature.

The reason is that the important information is already captured by:

1. the daily Close path, and
2. the High/Low excursion relative to the previous Close.

Therefore:

- **Close** measures directional trajectory.
- **High / Low** measure intraday price travel.
- **Previous Close** allows gaps and cross-day movement to be included.
- **Open** does not currently contribute a sufficiently unique regime property.

This does **not** mean Open is useless in general trading research. It means it is unnecessary for this specific monthly trend-vs-sideways definition.

---

# 4. Design Principle

A trending month should satisfy three different conditions:

1. Price should not wander excessively relative to the total range it achieves.
2. Daily returns should be coherently directional and distributed across the month.
3. The sequence of daily closes should remain temporally monotonic.

These produce three independent measurements:

\[
\boxed{
T_{\text{range}},
\quad
T_{\text{direction}},
\quad
T_{\text{mono}}
}
\]

Each is normalized approximately to:

\[
[0,1]
\]

where higher values mean more trend-like behavior.

---

# 5. Component 1 — Range Efficiency

## Purpose

This component answers:

> How much total price travel was required to create the month's final trading range?

A clean trend should expand its overall range efficiently.

A sideways or choppy month repeatedly traverses the same region, generating large cumulative movement without proportionally expanding the month's total range.

## 5.1 True Range

For every D1 bar:

\[
TR_t
=
\max(H_t,C_{t-1})
-
\min(L_t,C_{t-1})
\]

This is equivalent to the standard True Range definition.

It captures:

- normal intraday range,
- upside gaps,
- downside gaps.

## 5.2 Monthly Price Envelope

Let \(C_0\) be the final Close immediately before the calendar month begins.

Define:

\[
R_m
=
\max(C_0,H_1,H_2,\ldots,H_N)
-
\min(C_0,L_1,L_2,\ldots,L_N)
\]

This is the full price span reached during the month, including the initial starting price.

## 5.3 Range Efficiency Formula

Define:

\[
\boxed{
T_{\text{range}}
=
1-
\frac{
\ln
\left(
\frac{\sum_{t=1}^{N}TR_t}{R_m}
\right)
}{
\ln(N)
}
}
\]

Then clip numerically to:

\[
\boxed{0 \le T_{\text{range}} \le 1}
\]

This is closely related to an inverted normalized Choppiness-style measure.

Interpretation:

\[
T_{\text{range}}\rightarrow1
\]

means price expanded efficiently.

\[
T_{\text{range}}\rightarrow0
\]

means cumulative movement was large relative to the range ultimately achieved.

## Example

Suppose two months both travel from roughly 2500 to 2700.

### Month A

Total True Range:

\[
\sum TR=300
\]

Monthly envelope:

\[
R_m=220
\]

Then cumulative travel is only moderately larger than the total range.

This is relatively efficient.

### Month B

Total True Range:

\[
\sum TR=1000
\]

Monthly envelope:

\[
R_m=220
\]

Price has travelled through the same region repeatedly.

This is much more characteristic of choppy behavior.

---

# 6. Component 2 — Directional Coherence

## Purpose

This component answers:

> Were the month's daily movements consistently contributing to one directional move?

It should also penalize months where almost the entire directional movement occurs on only one or two exceptional days.

## 6.1 Daily Log Returns

Use Close-to-Close logarithmic returns:

\[
\boxed{
r_t
=
\ln
\left(
\frac{C_t}{C_{t-1}}
\right)
}
\]

Log returns are preferred because they are additive across time:

\[
\sum_{t=1}^{N}r_t
=
\ln
\left(
\frac{C_N}{C_0}
\right)
\]

## 6.2 Directional Coherence Formula

Define:

\[
\boxed{
T_{\text{direction}}
=
\frac{
\left|
\sum_{t=1}^{N}r_t
\right|
}{
\sqrt{
N
\sum_{t=1}^{N}r_t^2
}
}
}
\]

This is bounded by:

\[
\boxed{0\le T_{\text{direction}}\le1}
\]

because of the Cauchy-Schwarz inequality.

## Interpretation

### Perfectly consistent trend

If every daily return is identical and has the same sign:

\[
r_1=r_2=\cdots=r_N
\]

then:

\[
T_{\text{direction}}=1
\]

### Sideways / alternating movement

If returns repeatedly cancel each other:

\[
+1\%,-1\%,+1\%,-1\%,\ldots
\]

then:

\[
\sum r_t\approx0
\]

and therefore:

\[
T_{\text{direction}}\approx0
\]

## Important Property — Single Giant Jump

Suppose the month contains 20 trading days and only one day has a meaningful move.

For example:

```text
0
0
0
0
+10%
0
0
...
```

Then:

\[
T_{\text{direction}}
=
\frac{1}{\sqrt{20}}
\approx0.224
\]

Importantly, the result remains:

\[
\frac{1}{\sqrt{20}}
\]

whether that isolated move is +5%, +10%, or +50%.

Therefore a single huge jump cannot make the month appear like a perfect persistent trend.

This fixes an important weakness of Kaufman's Efficiency Ratio.

## 6.3 Useful Decomposition

The same quantity can be written as:

\[
T_{\text{direction}}
=
\underbrace{
\frac{
|\sum r_t|
}{
\sum |r_t|
}
}_{\text{Directional consistency}}
\times
\underbrace{
\frac{
\sum |r_t|
}{
\sqrt{N\sum r_t^2}
}
}_{\text{Movement participation}}
\]

The first term measures how strongly movements agree in direction.

The second term measures whether movement is distributed across many days instead of concentrated into a few exceptional observations.

This is one reason the formula is useful for monthly regime detection.

---

# 7. Component 3 — Temporal Monotonicity

## Purpose

Directional coherence does not know the order in which returns occurred.

For example, the following two months can contain similar return distributions:

### Persistent staircase

```text
100
102
101
103
102
104
103
105
```

### Trend then reversal

```text
100
102
104
106
108
106
104
102
```

The second month should not be considered as strongly trending because its regime reverses halfway through the month.

An order-sensitive statistic is therefore needed.

## 7.1 Log Price Sequence

Define:

\[
p_t=\ln(C_t)
\]

Use:

\[
p_0,p_1,\ldots,p_N
\]

where \(p_0\) is the log of the previous month's final Close.

## 7.2 Kendall Tau

Calculate Kendall's rank correlation between time and log price:

\[
\tau_b
=
\tau_b
\left(
[0,1,\ldots,N],
[p_0,p_1,\ldots,p_N]
\right)
\]

Then ignore direction by taking the absolute value:

\[
\boxed{
T_{\text{mono}}
=
|\tau_b|
}
\]

Therefore:

\[
\boxed{0\le T_{\text{mono}}\le1}
\]

## Interpretation

A persistent bullish trend:

\[
\tau_b\rightarrow+1
\]

A persistent bearish trend:

\[
\tau_b\rightarrow-1
\]

Because bullish and bearish trends are both trends:

\[
T_{\text{mono}}
=
|\tau_b|
\rightarrow1
\]

Sideways or repeatedly reversing movement tends toward lower values.

---

# 8. Optional Single Monthly Trend Score

The three measurements should remain separately available.

However, for visualization or descriptive analysis, they can be summarized into one score.

Use the geometric mean:

\[
\boxed{
S_m
=
\sqrt[3]{
T_{\text{range}}
T_{\text{direction}}
T_{\text{mono}}
}
}
\]

or equivalently:

\[
\boxed{
S_m
=
\left(
T_{\text{range}}
T_{\text{direction}}
T_{\text{mono}}
\right)^{1/3}
}
\]

where:

\[
0\le S_m\le1
\]

## Why Geometric Mean?

The arithmetic mean would allow one strong property to compensate excessively for one weak property.

Example:

\[
T_{\text{range}}=0.90
\]

\[
T_{\text{direction}}=0.90
\]

\[
T_{\text{mono}}=0.10
\]

Arithmetic mean:

\[
\frac{0.90+0.90+0.10}{3}
=
0.633
\]

This is too generous.

Geometric mean:

\[
(0.90\times0.90\times0.10)^{1/3}
\approx0.433
\]

The weak monotonicity meaningfully reduces the final score.

That is desirable because a strong trend should satisfy **all three conditions**, not merely one or two.

---

# 9. Do Not Use the Single Score as the Main Classifier

Although \(S_m\) is convenient, compressing three features into one inevitably discards information.

The preferred regime representation is therefore:

\[
\boxed{
X_m
=
[
T_{\text{range}},
T_{\text{direction}},
T_{\text{mono}}
]
}
\]

Use \(S_m\) mainly for:

- visualization,
- ranking,
- diagnostic plots,
- human-readable summaries.

Use the three-dimensional feature vector for the actual statistical regime model.

---

# 10. Regime Classification

Do **not** define an arbitrary threshold such as:

```text
score >= 0.50 -> Trend
score < 0.50  -> Sideways
```

A threshold such as 0.50 would simply become another manually chosen or optimized parameter.

Instead, let the historical distribution identify the regimes.

## 10.1 Two-Regime Mixture

Collect all historical months:

\[
X_1,X_2,\ldots,X_M
\]

where:

\[
X_m
=
[
T_{\text{range}},
T_{\text{direction}},
T_{\text{mono}}
]
\]

Fit a two-component probabilistic mixture model.

Conceptually:

\[
p(X)
=
\pi_T f_T(X)
+
\pi_S f_S(X)
\]

where:

- \(f_T\) is the trend-regime component,
- \(f_S\) is the sideways-regime component,
- \(\pi_T\) and \(\pi_S\) are their prior probabilities.

The component with larger values across the trend dimensions becomes the **Trend** component.

The other becomes **Sideways / Choppy**.

## 10.2 Soft Classification

For every historical month, calculate:

\[
\boxed{
q_m
=
P(\text{Trend}\mid X_m)
}
\]

Then:

\[
P(\text{Sideways}\mid X_m)
=
1-q_m
\]

Example:

```text
2018-04

T_range       = 0.79
T_direction   = 0.83
T_mono        = 0.88

Trend Score   = 0.83
P(Trend)      = 0.94
P(Sideways)   = 0.06
```

Another month:

```text
2019-04

T_range       = 0.31
T_direction   = 0.18
T_mono        = 0.29

Trend Score   = 0.25
P(Trend)      = 0.07
P(Sideways)   = 0.93
```

An ambiguous month can legitimately produce:

```text
P(Trend) = 0.51
```

instead of being forced into a hard binary label.

---

# 11. Calendar-Month Seasonality

Once every historical month has a trend probability, group observations by month-of-year.

For January:

\[
q_{\text{Jan},1},
q_{\text{Jan},2},
\ldots,
q_{\text{Jan},Y}
\]

where \(Y\) is the number of years in the historical sample.

The simplest soft estimate is:

\[
\boxed{
P(\text{Trend}\mid\text{January})
=
\frac{1}{Y}
\sum_{y=1}^{Y}
q_{\text{Jan},y}
}
\]

Likewise for every other month.

Then:

\[
\boxed{
P(\text{Sideways}\mid m)
=
1-P(\text{Trend}\mid m)
}
\]

---

# 12. Bayesian Shrinkage / Small-Sample Adjustment

Calendar-month seasonality has an unavoidable sample-size problem.

Even 20 years of history gives only:

```text
20 Januaries
20 Februaries
20 Marches
...
```

Therefore raw probabilities should not be interpreted as exact frequencies.

## 12.1 Hard Binary Labels

If hard labels are ever used, with:

- \(k_m\) trending observations,
- \(N_m\) total observations,

a simple Jeffreys-prior estimate is:

\[
\boxed{
P_m
=
\frac{
k_m+0.5
}{
N_m+1
}
}
\]

## 12.2 Soft Probabilities

If the regime model provides:

\[
q_{m,y}
=
P(\text{Trend})
\]

for each observation, an analogous smoothed estimator is:

\[
\boxed{
P_m
=
\frac{
0.5+\sum_y q_{m,y}
}{
N_m+1
}
}
\]

A more complete implementation should also calculate uncertainty intervals rather than reporting only one probability.

---

# 13. Final Desired Output

The final research result should eventually look approximately like:

| Month | P(Trend) | P(Sideways) | Sample | Uncertainty |
|---|---:|---:|---:|---:|
| January | 68% | 32% | 20 | ... |
| February | 44% | 56% | 20 | ... |
| March | 72% | 28% | 20 | ... |
| April | 38% | 62% | 20 | ... |
| May | 51% | 49% | 20 | ... |
| June | ... | ... | ... | ... |
| July | ... | ... | ... | ... |
| August | ... | ... | ... | ... |
| September | ... | ... | ... | ... |
| October | ... | ... | ... | ... |
| November | ... | ... | ... | ... |
| December | ... | ... | ... | ... |

This represents:

\[
P(\text{market regime} \mid \text{calendar month})
\]

not expected return.

---

# 14. Why Several Alternatives Were Rejected

## 14.1 Kaufman Efficiency Ratio

Standard efficiency ratio:

\[
ER
=
\frac{
|C_N-C_0|
}{
\sum_{t=1}^{N}|C_t-C_{t-1}|
}
\]

The major failure case is a month with one giant jump and otherwise flat prices.

Example:

```text
0
0
0
+10%
0
0
...
```

The numerator and denominator can become approximately identical:

\[
ER\approx1
\]

which classifies the month as a perfect trend.

That is undesirable.

The proposed directional coherence measure instead gives:

\[
T_{\text{direction}}
=
\frac{1}{\sqrt N}
\]

for a single isolated move.

## 14.2 ADX

ADX should not define the regime because:

- it introduces a lookback parameter,
- it introduces threshold choices,
- the result becomes "seasonality of ADX" rather than a fundamental characterization of price behavior.

ADX may still be used later as an external comparison.

## 14.3 Hurst Exponent

Hurst estimation is unattractive here because a calendar month contains only approximately 20–23 daily observations.

That is an extremely small sample for reliable Hurst estimation.

Therefore it adds substantial estimation noise without solving a unique problem that the other measurements do not already address.

## 14.4 Linear Regression \(R^2\)

Regression fit can be useful diagnostically, but it can behave poorly when:

- a structural jump occurs,
- the month consists of multiple regimes,
- price trends and later reverses.

It is therefore not necessary in the core formula.

## 14.5 Moving Averages

Moving-average slope or ordering introduces arbitrary lookback lengths.

Examples:

- SMA10,
- EMA20,
- SMA5 vs SMA20.

These create unnecessary parameters and partly smooth away the behavior we are attempting to measure.

## 14.6 Heikin-Ashi

Heikin-Ashi deliberately smooths OHLC data.

Because the objective is to determine whether the **raw market itself** was smooth/trending or choppy, the classifier should not receive a price transformation specifically designed to visually emphasize trends.

## 14.7 Candle Body / Wick Features

Candidate features such as:

\[
\frac{|C-O|}{H-L}
\]

bullish-candle percentage,

wick ratios,

close location within range,

and average candle body were considered.

They are currently excluded because much of their regime information is already captured by:

- cumulative True Range,
- monthly envelope,
- Close-to-Close returns,
- return concentration,
- temporal ordering.

Adding many candle-morphology features would make the definition increasingly arbitrary and could encourage overfitting.

They may later be tested as secondary explanatory variables, but should not be part of the initial regime definition.

---

# 15. Avoid Strategy Information in the Formula

Do not include:

- strategy Profit Factor,
- strategy Sharpe,
- strategy return,
- win rate,
- breakout success rate,
- trade count,
- EA drawdown,
- EA expectancy.

The regime classifier must be independent of the strategy.

Otherwise the analysis becomes circular:

```text
good strategy month
    ->
classified as trend
    ->
strategy performs well in trend
```

Instead:

```text
raw market prices
    ->
regime classification
    ->
calendar regime seasonality
    ->
strategy tested conditionally afterward
```

This separation is important.

---

# 16. Data Consistency

## D1 Candle Session

Daily OHLC values depend on the data vendor's daily-session boundary.

Different brokers may construct different D1 candles from the same underlying intraday market.

Therefore the entire history should use one consistent D1 definition.

Preferably:

\[
\boxed{\text{New York-close style D1 data}}
\]

if reliable data is available.

The exact convention matters less at a monthly horizon than it would on M15, but mixing candle-session definitions across the dataset should still be avoided.

---

# 17. Edge Cases

## 17.1 Missing Trading Days

Use the actual number of available trading bars:

\[
N=\text{number of D1 observations in the month}
\]

Do not force every month to contain the same number of bars.

## 17.2 Zero Monthly Range

If:

\[
R_m=0
\]

the range-efficiency formula is undefined.

This should be treated as an invalid / degenerate observation rather than assigned an arbitrary trend score.

For liquid XAUUSD or major FX data this should practically never occur.

## 17.3 Zero Return Energy

If:

\[
\sum r_t^2=0
\]

then the entire month was mathematically unchanged.

Set:

\[
T_{\text{direction}}=0
\]

because there was no trend.

## 17.4 Numerical Clipping

Floating-point arithmetic may occasionally produce values marginally outside the intended range.

Therefore use:

\[
T_i
\leftarrow
\min(1,\max(0,T_i))
\]

for normalized components.

---

# 18. Complete Formula Summary

For each calendar month containing \(N\) D1 bars:

## Step 1 — True Range

\[
TR_t
=
\max(H_t,C_{t-1})
-
\min(L_t,C_{t-1})
\]

## Step 2 — Monthly Envelope

\[
R_m
=
\max(C_0,H_1,\ldots,H_N)
-
\min(C_0,L_1,\ldots,L_N)
\]

## Step 3 — Range Efficiency

\[
\boxed{
T_{\text{range}}
=
1-
\frac{
\ln
\left(
\frac{\sum TR_t}{R_m}
\right)
}{
\ln(N)
}
}
\]

## Step 4 — Log Returns

\[
r_t
=
\ln
\left(
\frac{C_t}{C_{t-1}}
\right)
\]

## Step 5 — Directional Coherence

\[
\boxed{
T_{\text{direction}}
=
\frac{
|\sum r_t|
}{
\sqrt{
N\sum r_t^2
}
}
}
\]

## Step 6 — Log Prices

\[
p_t=\ln(C_t)
\]

## Step 7 — Temporal Monotonicity

\[
\boxed{
T_{\text{mono}}
=
|\tau_b(t,p_t)|
}
\]

## Step 8 — Descriptive Trend Score

\[
\boxed{
S_m
=
\left(
T_{\text{range}}
T_{\text{direction}}
T_{\text{mono}}
\right)^{1/3}
}
\]

## Step 9 — Regime Feature Vector

\[
\boxed{
X_m
=
[
T_{\text{range}},
T_{\text{direction}},
T_{\text{mono}}
]
}
\]

## Step 10 — Probabilistic Classification

Fit a two-component regime model:

\[
\boxed{
q_m
=
P(\text{Trend}\mid X_m)
}
\]

## Step 11 — Calendar-Month Seasonality

For calendar month \(j\):

\[
\boxed{
P(\text{Trend}\mid j)
=
\frac{1}{N_j}
\sum_y q_{j,y}
}
\]

with statistical shrinkage / uncertainty reporting.

Finally:

\[
\boxed{
P(\text{Sideways}\mid j)
=
1-P(\text{Trend}\mid j)
}
\]

---

# 19. Recommended Research Pipeline

```text
Raw D1 OHLC
    |
    v
Create calendar-month groups
    |
    +--> High / Low / Previous Close
    |       |
    |       v
    |   T_range
    |
    +--> Daily Close
            |
            +--> Log returns
            |       |
            |       v
            |   T_direction
            |
            +--> Ordered log-price path
                    |
                    v
                T_mono

[T_range, T_direction, T_mono]
            |
            +--> Geometric mean -> S_m
            |                       |
            |                       v
            |                  diagnostics only
            |
            v
Two-regime probabilistic model
            |
            v
P(Trend | each historical month)
            |
            v
Group by January ... December
            |
            v
Seasonality probability
            |
            v
P(Trend | January)
P(Trend | February)
...
P(Trend | December)
```

---

# 20. Current Recommended Definition

The current preferred definition of a trending month is:

> A month whose overall price range expands efficiently, whose daily Close-to-Close movements are coherently directional and distributed across the month, and whose daily closing path remains persistently monotonic through time.

The corresponding feature vector is:

\[
\boxed{
[
T_{\text{range}},
T_{\text{direction}},
T_{\text{mono}}
]
}
\]

This is preferable to a single technical indicator because each component captures a different failure mode:

| Component | Main problem detected |
|---|---|
| \(T_{\text{range}}\) | intraday wandering / repeated range traversal |
| \(T_{\text{direction}}\) | cancellation and movement concentrated in a few days |
| \(T_{\text{mono}}\) | trend reversal or poor temporal persistence |

Together they provide a compact, parameter-light definition of monthly market regime.

---

# 21. Status

This document describes the **current research specification**, not a permanently fixed formula.

The next correct step is empirical validation.

Before accepting the model as final, it should be tested against synthetic and historical examples representing at least:

1. perfectly smooth bullish trend,
2. perfectly smooth bearish trend,
3. quiet sideways market,
4. volatile sideways market,
5. one-day price shock,
6. trend followed by reversal,
7. staircase trend,
8. trend with large intraday wicks,
9. random walk,
10. mixed / ambiguous month.

The formula should only be considered validated if its rankings across these controlled cases agree with the intended economic meaning of **trend versus sideways**.
