# TOP — Fix Report

What changed, why, and the honest numbers before/after. Read this before trusting
any output from the system.

## 1. Data leakage removed

| Model | What leaked | Before | After |
|---|---|---|---|
| B14 Acceleration Capability | `is_boundary`/`is_dot_ball` were direct functions of `batter_runs`, which **is** the target | R² = 0.92 | R² = 0.15 |
| B15 Death-Over Optimization | same pattern; target also includes the current ball's own runs | (not previously reported) | R² = 0.59 |
| B5 Strike Rotation | `is_boundary`/`is_dot_ball` ruled the `[1,3,5]`-runs target in or out for the same ball | (not previously reported) | ROC-AUC = 0.749 |
| B13 Wicket-Loss Mitigation | `is_boundary` was the *current* ball's own outcome sitting next to `is_wicket`, also the current ball | (not previously reported) | ROC-AUC = 0.735 (now uses the *previous* ball's boundary as a momentum feature instead) |

B14's R² dropping from 0.92 to 0.15 is the clearest single piece of evidence that the
leakage was real and the fix worked — 0.15 is a believable number for "predict the
exact runs off one specific ball in the death overs from pre-ball context"; 0.92 was not.

## 2. Split methodology made consistent everywhere

Every model that reports a train/test metric now uses the same `match_level_split()`
(shuffle unique `match_id`s, seed 42, 80/20) — no model's test set can contain rows
from a match any of its own training rows came from.

- **Already correct, unchanged methodology:** W1–W4, W6–W9, W11, W13, W15, B1, B2, B3, B5
- **Fixed (was a plain/sequential row split):** B6, B10, B11, B13, B14, B15
- **Fixed (had *no* train/test split at all — trained and "evaluated" on the same rows):** B8, B9, B12

B12 (Gap Analysis/KNN) had an additional issue: it fit on the first 30,000 rows and then
predicted on the *entire* dataset, including those same 30,000 rows. It now fits on a
train-match sample and is evaluated only on held-out test-match rows it never saw.

## 3. W12 (Bowler Form Baseline) — test-set leakage in a lookup table

The original built each bowler's "career" stats from the **whole** dataset, including
matches that were supposedly held out for testing — so a bowler's test-set performance
was already baked into the number being evaluated against. Rebuilt to compute the
lookup from training matches only, with a documented fallback (global average) for any
bowler never seen in training. Sanity-checked via the correlation between the
train-only lookup value and each bowler's actual future economy in the test set
(0.13 — weak but real signal, correctly modest for a coarse "career average" proxy).

## 4. W14 (Economy Trend Analysis) — redesigned, not just retrained

The original fit a single global ARIMA(2,1,2) model across every bowler's economy
figure concatenated into one sequence, ignoring bowler identity entirely, then
forecast forward from wherever that arbitrary sequence ended. Multi-step ARIMA
forecasts flatten toward the series mean, which is exactly why the saved predictions
had ~zero variance (std ≈ 0.000158) — the model was structurally incapable of saying
anything bowler-specific, and can't be wired into a live per-ball system anyway (it
has no notion of "predict for bowler X right now").

Replaced with a supervised regression using the same snapshot-feature template as
every other bowling model. Prediction std is now 2.27 (real variation across bowlers/
situations). R² is **-0.12** — worse than predicting the mean. This is disclosed
honestly rather than hidden: short-horizon economy "trend" appears to be close to
unpredictable from the features available here. Its reliability weight in the engine
(0.20, the floor) reflects that — the aggregator now trusts this signal least of all
28 models, instead of exactly as much as everything else.

## 5. W9 vs W15 — were producing byte-identical output

Both had the exact same target formula, the exact same features, and the exact same
algorithm (LightGBM) — "Death Over Accuracy" and "Powerplay Containment" were the same
model wearing two names. Fixed by phase-filtering: W9 now trains only on the over=15
timeout snapshot (right before the death overs), W15 only on the over=5 snapshot (end
of powerplay). Their numbers are now genuinely different (Accuracy 0.642 vs 0.710).

W8 (Yorker Effectiveness) shared the same formula too (just a different algorithm,
CatBoost) — changed its target to boundary-prevention only, dropping the economy
condition, so it measures something distinct.

## 6. Dataset limitations that no amount of retraining fixes

`dataset.csv` has no bowler-type (pace/spin), no delivery-type (yorker/bouncer/slower
ball), and no batter-handedness field. This means:

- **W7 Spin Control**, **W8 Yorker Effectiveness**, **W9 Death Over Accuracy** can only
  measure "is the bowler's economy/boundary rate improving," not the specific tactic
  their names claim, because the tactic itself isn't recorded anywhere in the data.
- **W13 L/R Matchup Bias** can't see handedness, so it measures general spell-improvement
  + wicket-taking bias, not literal left/right-arm or left/right-hand matchups.
- **B9 Spin Vulnerability** is really phase-based wicket probability generally.

These are labeled with a `note` field in `model_config.json` rather than silently
presented as more specific than they are.

## 7. Decision engine changes

- **Reliability is now data-driven.** `training/build_reliability.py` turns each
  model's real validation metric (ROC-AUC for classifiers, R² for regressors, a
  correlation for W12, a documented neutral 0.7 for the two hand-tuned formula
  "models" W5/W10) into a 0.15–1.6 multiplier in `reliability.json`, which the
  Weighter now loads instead of trusting every model at a flat 1.0.
- **Confidence score rewritten.** It was `0.5 + |score|/6` — an arbitrary bounded
  formula with no connection to actual certainty. It's now based on (a) the margin
  between the chosen action and the runner-up, and (b) what fraction of the model
  roster actually had a signal for this ball. A near-tie between the top two actions
  now correctly reads as lower-confidence, regardless of the raw score's magnitude.
- **RuleValidator returns a full audit trail** (`audit` field) — every ranked action's
  score and blocked-status, not just the winner, so a coach (or this frontend) can see
  why the runner-up wasn't picked.
- **Action-direction judgment call documented, not silently guessed.** B1/B10/B11 all
  map "accelerate: -1" (a high projected score argues *against* accelerating further).
  That's a defensible reading, not a verified one — it's called out explicitly in
  `model_config.json`'s `action_direction_note` so it can be revisited.

## 8. What was NOT changed

- B7 (Matchup Matrix) is a batter×bowler lookup table, not a per-ball predictive
  model. It never fed the decision engine and was left out of scope.
- W5 and W10 are fixed-weight linear scoring formulas, not trained models. They're
  kept as formulas (that's a legitimate design choice for a hand-tuned tactical
  score) but are now labeled as such in their saved artifact and given a documented
  neutral reliability instead of an invented accuracy number.

## Full metrics table (all real, from the retrained models)

| Model | Label | Metric |
|---|---|---|
| W1 | Economy Predictor | R² 0.648 |
| W2 | Wicket Probability Predictor | AUC 0.607 |
| W3 | Dot Ball Pressure | R² 0.679 |
| W4 | Variation Control | AUC 0.740 |
| W5 | Bowling Change Optimization | formula (not a fit model) |
| W6 | Line & Length Consistency | AUC 0.586 |
| W7 | Spin Control | AUC 0.782 |
| W8 | Yorker Effectiveness | AUC 0.755 |
| W9 | Death Over Accuracy | AUC 0.650 |
| W10 | Field-Set Optimization | formula (not a fit model) |
| W11 | Run-Containment | R² 0.266 |
| W12 | Bowler Form / Baseline | lookup, corr 0.134 |
| W13 | L/R Matchup Bias | AUC 0.622 |
| W14 | Economy Trend Analysis | R² -0.124 |
| W15 | Powerplay Containment | AUC 0.772 |
| B1 | Run Projection | R² 0.358 |
| B2 | Dismissal Risk | Cox survival (hazard ratios) |
| B3 | Shot Aggression | Acc 0.452 (3-class, baseline 0.333) |
| B5 | Strike Rotation | AUC 0.749 |
| B6 | Partnership Stability | R² 0.511 |
| B8 | Powerplay Exploitation | AUC 0.565 |
| B9 | Spin Vulnerability | AUC 0.602 |
| B10 | Scoring Velocity | R² 0.056 |
| B11 | Targeted Run-Rate | R² 0.470 |
| B12 | Gap Analysis | AUC 0.553 |
| B13 | Wicket-Loss Mitigation | AUC 0.735 |
| B14 | Acceleration Capability | R² 0.149 |
| B15 | Death-Over Optimization | R² 0.586 |

Some of these are weak (W14, B10, W11, B3, W6, B8, B12) — that's disclosed, not
smoothed over, and is exactly what the reliability weighting is for.
