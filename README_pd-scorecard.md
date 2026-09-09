# PD Scorecard — Give Me Some Credit

An application-stage probability-of-default scorecard built on the Kaggle *Give Me Some Credit* dataset (~150K consumer credit accounts), taken through the full lifecycle a scorecard actually goes through in production: WOE binning, logistic regression, points scaling, **calibration diagnostics at the decile level**, isotonic recalibration, and a marginal break-even cutoff analysis that answers the question the model itself does not — *where should the approve/decline line sit?*

The headline result is not the AUC. It is what the calibration diagnostic found underneath it.

---

## The finding

The model passes every aggregate calibration check. Mean predicted PD matches the observed default rate on the holdout to within a few basis points, and the Hosmer–Lemeshow style bucket comparison looks clean at first glance.

It is still wrong where it matters most.

Predicted probabilities are **under-dispersed** — the distribution is compressed toward the base rate relative to the observed outcome. In the safest deciles this shows up as roughly **2x over-prediction of risk**: the model assigns materially more default probability to the best accounts than they actually carry. Because the same compression over-predicts safety at the risky end, the two errors offset and the aggregate calibration statistic never flags it.

This matters commercially, not academically. The safest deciles are the accounts you compete hardest to win. Over-stating their risk means over-pricing them, and over-pricing your best applicants is how a portfolio adversely selects itself.

The fix implemented here is **isotonic regression recalibration** fitted on a held-out calibration split — monotone, non-parametric, and preserves the model's rank ordering (so AUC/Gini/KS are unchanged) while correcting the probability level decile by decile. Platt scaling was rejected because a single-parameter sigmoid cannot correct a shape error of this kind.

**Reproduce it:** `python -m src.calibration` writes `outputs/calibration_by_decile.csv`, with predicted vs. observed PD before and after recalibration.

---

## Results

| Metric | Holdout |
|---|---|
| AUC | 0.849 |
| Gini | 0.699 |
| KS | 0.541 |
| Approval rate at chosen cutoff | 79.7% |
| Share of defaults screened out at that cutoff | 69% |

Discrimination metrics are reported on the holdout split, after WOE transformation fitted on train only.

A note on the last two rows, because this is where scorecard write-ups usually overclaim: approving 79.7% of applicants and excluding 69% of the defaulters in the population **is not** a 69% reduction in the default rate of the approved book. The approved book still contains the 31% of defaulters that scored above the line, spread over a smaller denominator. The honest statement is the one in the table.

---

## Cutoff selection

Discrimination tells you the ranking is good. It does not tell you where to cut. The cutoff here is set by marginal economics rather than by picking a round approval rate.

For each score band, the pipeline computes expected marginal profit:

```
E[profit | approve band] = (1 - PD_band) × margin × EAD  −  PD_band × LGD × EAD
```

which is positive while `PD_band < margin / (margin + LGD)`.

Sweeping the bands produces a **zero-crossing at score band 580–584** — the marginal band where approving one more applicant stops adding profit.

The more useful output is the shape around it. Total portfolio P&L is **flat across bands 575–585**, a plateau rather than a peak. Practically:

- Precision in cutoff placement is worth less than the modelling effort usually spent on it — anywhere in that window is economically equivalent.
- The binding constraint on where you actually sit inside the plateau is volume appetite and risk tolerance, not the model.
- Cutoffs *outside* the plateau are where the P&L falls away quickly, so the analysis is worth doing even though the answer is a range.

Assumptions (`margin`, `LGD`, `EAD`) are parameters in `src/cutoff_analysis.py`, not constants buried in the code, so the plateau can be re-derived under different economics. The plateau width is itself sensitive to the margin/LGD ratio — worth stating in interview rather than presenting the range as a property of the data.

---

## Data handling

Source: [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) (Kaggle, `cs-training.csv`). Target: `SeriousDlqin2yrs` — 90+ days past due within two years. Base rate ~6.7%.

Data is **not committed to this repo.** See `data/README.md` for the download and expected filename.

Cleaning decisions that are not obvious from the column names:

- **Sentinel codes 96 and 98** in the three delinquency-count columns (`NumberOfTime30-59DaysPastDueNotWorse`, `NumberOfTime60-89DaysPastDueNotWorse`, `NumberOfTimes90DaysLate`) are not counts. They are administrative codes and appear on the same rows across all three columns. Treated as a separate category rather than winsorized into the tail, because the rows carry a distinctly elevated default rate and folding them into "many delinquencies" loses that signal.
- **`MonthlyIncome`** is missing on ~20% of rows and **`NumberOfDependents`** on ~2.6%. Missingness is handled as its own WOE bin rather than imputed — for application scoring, "income not supplied" is itself predictive and is available at decision time.
- **`RevolvingUtilizationOfUnsecuredLines`** and **`DebtRatio`** contain values in the thousands, which are not plausible ratios. Capped at the 99.5th percentile before binning.
- **`age`** has a record at 0. Dropped.

---

## Method

1. **Binning** — supervised binning with a monotonicity constraint on the WOE trend, minimum 5% of population per bin. Monotonicity is enforced because a scorecard whose points move non-monotonically in a driver is not defensible to a credit committee even when it fits better.
2. **Feature selection** — Information Value filter (retain 0.02 ≤ IV ≤ 0.5; the upper bound catches leakage-like dominance), then correlation pruning on the WOE-transformed matrix.
3. **Model** — logistic regression on WOE values, L2 regularised. Chosen over a gradient boosting model deliberately: the artefact has to be explainable to a regulator and convertible to a points table, and on this dataset the AUC gap to a tuned GBM is small enough that it does not buy back the loss in explainability.
4. **Scaling** — points allocation at PDO = 20, base score 600 at 50:1 odds.
5. **Calibration** — decile-level predicted-vs-observed, then isotonic recalibration on a separate calibration split.
6. **Scoring & monitoring** — scored records loaded to SQLite; decile assignment, KS, and PSI computed in SQL (`sql/`).

## SQL

`sql/` holds the scoring and monitoring layer, run against SQLite rather than in pandas on purpose — decile assignment, gains tables, and stability monitoring are things that live in a warehouse in any real deployment.

- `01_create_schema.sql` — scored-account table and indexes
- `02_decile_scoring.sql` — `NTILE(10)` banding with per-decile bad rate, cumulative capture, and lift
- `03_ks_gains_table.sql` — KS statistic and the full gains table in SQL
- `04_psi_monitoring.sql` — Population Stability Index, expected vs. actual distribution

## Running it

```bash
pip install -r requirements.txt
# place cs-training.csv in data/ — see data/README.md
python -m src.pipeline          # end-to-end: clean → bin → fit → score
python -m src.calibration       # decile calibration + isotonic fix
python -m src.cutoff_analysis   # break-even sweep and P&L plateau
sqlite3 outputs/scorecard.db < sql/02_decile_scoring.sql
```

Outputs land in `outputs/` (gitignored).

## Repo layout

```
src/
  data_prep.py        cleaning, sentinel handling, missingness policy
  binning.py          monotonic WOE binning, IV computation
  model.py            logistic regression, points scaling
  calibration.py      decile calibration diagnostic + isotonic recalibration
  cutoff_analysis.py  marginal break-even sweep, P&L curve
  pipeline.py         orchestration
sql/                  scoring, gains, KS, PSI
notebooks/            exploratory work
data/README.md        how to obtain the data
```

## What I would do differently

- **Reject inference is absent.** The dataset is an accepted-population sample, so the model is trained on a biased selection. Without a reject sample, parcelling or a bureau-score-anchored augmentation is guesswork — but the limitation should be stated rather than ignored, because it means the PD estimates are conditional on historical approval policy.
- **No time dimension.** The data has no origination date, so this is a through-the-cycle-flavoured model with no PIT/TTC distinction available and no vintage analysis possible. PSI monitoring in `sql/` is built against a synthetic split for that reason.
- **Single-segment.** A production application scorecard would almost certainly segment (thin-file vs. thick-file at minimum) and fit separate models.
