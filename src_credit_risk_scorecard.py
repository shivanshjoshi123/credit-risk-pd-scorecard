"""
Consumer Credit Risk: PD Scorecard & Approval Strategy
------------------------------------------------------
Hybrid SQL + Python pipeline mirroring a production credit risk workflow:

    SQL   -> data profiling, portfolio segmentation, decile analysis,
             score-band aggregation, cutoff strategy
    Python-> model fitting only (logistic regression PD model)

Data is held in a relational store (SQLite) throughout. Scores are written
back to the database and all reporting is done via SQL window functions
and grouped aggregations, as it would be against a bank's risk data mart.

Usage:
    python credit_risk_scorecard_sql.py --data cleaned_data.csv --out ./output
"""

import argparse
import os
import sqlite3

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

TARGET = "SeriousDlqin2yrs"

# Score scaling.
#
# IMPORTANT: this is an INTERNAL scorecard scale, calibrated here purely from
# the model's own log-odds. It is NOT a bureau score - the values are not
# comparable to CIBIL (300-900), FICO (300-850) or any external scale, and a
# score of e.g. 580 here carries no relationship to 580 on a bureau scale.
# The anchor below is an arbitrary choice; changing it shifts every score
# without changing the model, the ranking, or any decision.
BASE_SCORE = 600            # score at which good:bad odds = BASE_ODDS
POINTS_TO_DOUBLE_ODDS = 20  # PDO - every +20 points doubles the good:bad odds
BASE_ODDS = 20              # 20 good accounts per 1 bad account at BASE_SCORE

# Illustrative unit economics for the expected-value cutoff strategy.
# In production these would be calibrated from product margin and LGD.
GOOD_ACCOUNT_PROFIT = 100
BAD_ACCOUNT_LOSS = 1000

TEST_SIZE = 0.30
RANDOM_STATE = 42


# --------------------------------------------------------------------------
# 1. Load raw data into the database
# --------------------------------------------------------------------------

def build_database(csv_path, conn):
    """Load the source file into SQLite and index it for querying."""
    df = pd.read_csv(csv_path)
    # Column names with hyphens are awkward in SQL - normalise to snake-safe.
    df.columns = [c.replace("-", "_") for c in df.columns]
    df["account_id"] = np.arange(1, len(df) + 1)
    df.to_sql("applications", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX idx_app_id ON applications(account_id)")
    conn.commit()
    return df


# --------------------------------------------------------------------------
# 2. Portfolio profiling in SQL
# --------------------------------------------------------------------------

PROFILE_SQL = """
SELECT
    COUNT(*)                                        AS total_accounts,
    SUM(SeriousDlqin2yrs)                           AS bad_accounts,
    ROUND(AVG(SeriousDlqin2yrs), 4)                 AS portfolio_bad_rate,
    ROUND(AVG(age), 1)                              AS avg_age,
    ROUND(AVG(MonthlyIncome), 0)                    AS avg_monthly_income,
    ROUND(AVG(DebtRatio), 3)                        AS avg_debt_ratio
FROM applications;
"""

# Bad rate by delinquency history - the single strongest risk driver.
# Demonstrates segmentation logic done at the data layer, not in pandas.
SEGMENT_SQL = """
SELECT
    CASE
        WHEN NumberOfTimes90DaysLate = 0 THEN '0 - never 90+ DPD'
        WHEN NumberOfTimes90DaysLate = 1 THEN '1 occurrence'
        WHEN NumberOfTimes90DaysLate = 2 THEN '2 occurrences'
        ELSE '3+ occurrences'
    END                                             AS delinquency_segment,
    COUNT(*)                                        AS accounts,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct_of_portfolio,
    ROUND(AVG(SeriousDlqin2yrs), 4)                 AS bad_rate
FROM applications
GROUP BY delinquency_segment
ORDER BY bad_rate;
"""


# --------------------------------------------------------------------------
# 3. Fit the PD model (Python) and write scores back to the database
# --------------------------------------------------------------------------

def fit_pd_model(df):
    """Train a logistic regression PD model on a stratified holdout split."""
    feature_cols = [c for c in df.columns if c not in (TARGET, "account_id")]
    X = df[feature_cols]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("logistic_regression", LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        )),
    ])
    model.fit(X_train, y_train)

    scored = pd.DataFrame({
        "account_id": df.loc[X_test.index, "account_id"].values,
        "actual_bad": y_test.values,
        "predicted_pd": model.predict_proba(X_test)[:, 1],
    })
    return model, scored, y_train


def pd_to_score(pd_values):
    """Log-odds transform: PD -> internal scorecard points."""
    factor = POINTS_TO_DOUBLE_ODDS / np.log(2)
    offset = BASE_SCORE - factor * np.log(BASE_ODDS)
    odds = (1 - pd_values) / pd_values
    return np.round(offset + factor * np.log(odds), 0)


def write_scores(scored, conn):
    """Persist model output to the database for downstream SQL reporting."""
    scored.to_sql("test_scores", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX idx_score_id ON test_scores(account_id)")
    conn.commit()


# --------------------------------------------------------------------------
# 4. Model validation metrics
# --------------------------------------------------------------------------

def validation_metrics(scored):
    auc = roc_auc_score(scored["actual_bad"], scored["predicted_pd"])
    fpr, tpr, _ = roc_curve(scored["actual_bad"], scored["predicted_pd"])
    return {"auc": auc, "gini": 2 * auc - 1, "ks": float(np.max(tpr - fpr))}


# --------------------------------------------------------------------------
# 5. Risk-decile rank-ordering test - SQL window function
# --------------------------------------------------------------------------

DECILE_SQL = """
WITH ranked AS (
    SELECT
        account_id,
        actual_bad,
        predicted_pd,
        internal_score,
        NTILE(10) OVER (ORDER BY predicted_pd) AS risk_decile
    FROM test_scores
)
SELECT
    risk_decile,
    COUNT(*)                                        AS accounts,
    ROUND(AVG(predicted_pd), 4)                     AS avg_predicted_pd,
    ROUND(AVG(actual_bad), 4)                       AS observed_bad_rate,
    SUM(actual_bad)                                 AS bad_accounts,
    ROUND(MIN(internal_score), 0)                     AS min_score,
    ROUND(MAX(internal_score), 0)                     AS max_score
FROM ranked
GROUP BY risk_decile
ORDER BY risk_decile DESC;
"""

# Cumulative bad capture - what share of all defaulters sits in the
# riskiest N deciles. This is the Gini curve expressed in SQL.
GAIN_SQL = """
WITH ranked AS (
    SELECT
        actual_bad,
        NTILE(10) OVER (ORDER BY predicted_pd DESC) AS risk_band
    FROM test_scores
),
banded AS (
    SELECT
        risk_band,
        COUNT(*)        AS accounts,
        SUM(actual_bad) AS bads
    FROM ranked
    GROUP BY risk_band
)
SELECT
    risk_band,
    accounts,
    bads,
    ROUND(100.0 * SUM(accounts) OVER (ORDER BY risk_band)
          / SUM(accounts) OVER (), 2)               AS cum_pct_accounts,
    ROUND(100.0 * SUM(bads) OVER (ORDER BY risk_band)
          / SUM(bads) OVER (), 2)                   AS cum_pct_bads_captured
FROM banded
ORDER BY risk_band;
"""


# --------------------------------------------------------------------------
# 6. Approval cutoff strategy - SQL aggregation over score bands
# --------------------------------------------------------------------------

STRATEGY_SQL = f"""
WITH cutoffs(score_cutoff) AS (
    VALUES (560),(570),(580),(590),(600),(610),(620),(630),(640)
),
portfolio AS (
    SELECT COUNT(*) AS total_accounts FROM test_scores
)
SELECT
    c.score_cutoff,
    COUNT(t.account_id)                             AS approved_accounts,
    ROUND(1.0 * COUNT(t.account_id) / p.total_accounts, 4)
                                                    AS approval_rate,
    ROUND(AVG(t.actual_bad), 4)                     AS observed_bad_rate,
    ROUND(AVG(t.predicted_pd), 4)                   AS avg_predicted_pd,
    ROUND(SUM(
        (1 - t.predicted_pd) * {GOOD_ACCOUNT_PROFIT}
        - t.predicted_pd * {BAD_ACCOUNT_LOSS}
    ), 2)                                           AS total_expected_value,
    ROUND(SUM(
        (1 - t.predicted_pd) * {GOOD_ACCOUNT_PROFIT}
        - t.predicted_pd * {BAD_ACCOUNT_LOSS}
    ) / COUNT(t.account_id), 2)                     AS expected_value_per_account
FROM cutoffs c
CROSS JOIN portfolio p
LEFT JOIN test_scores t
       ON t.internal_score >= c.score_cutoff
GROUP BY c.score_cutoff, p.total_accounts
ORDER BY c.score_cutoff;
"""

BEST_CUTOFF_SQL = """
SELECT score_cutoff, approval_rate, observed_bad_rate, total_expected_value
FROM strategy_results
ORDER BY total_expected_value DESC
LIMIT 1;
"""


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="cleaned_data.csv",
                        help="Path to the cleaned source data CSV")
    parser.add_argument("--out", default="output",
                        help="Directory for output files")
    parser.add_argument("--db", default="credit_risk.db",
                        help="SQLite database file to build")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    db_path = os.path.join(args.out, args.db)
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)

    # --- 1. Load ---------------------------------------------------------
    print("Loading source data into SQLite...")
    df = build_database(args.data, conn)
    print(f"  loaded {len(df):,} accounts into table 'applications'\n")

    # --- 2. Profile (SQL) ------------------------------------------------
    print("=" * 70)
    print("PORTFOLIO PROFILE  [SQL]")
    print("=" * 70)
    print(pd.read_sql(PROFILE_SQL, conn).T.to_string(header=False))

    print("\nBad rate by delinquency history segment  [SQL]")
    print(pd.read_sql(SEGMENT_SQL, conn).to_string(index=False))

    # --- 3. Model (Python) ----------------------------------------------
    print("\n" + "=" * 70)
    print("PD MODEL  [Python]")
    print("=" * 70)
    model, scored, y_train = fit_pd_model(df)
    scored["internal_score"] = pd_to_score(scored["predicted_pd"].values)
    write_scores(scored, conn)
    print(f"  train bad rate: {y_train.mean():.4f}")
    print(f"  test  bad rate: {scored['actual_bad'].mean():.4f}")
    print(f"  scored {len(scored):,} holdout accounts -> table 'test_scores'")

    m = validation_metrics(scored)
    print(f"\n  AUC : {m['auc']:.3f}")
    print(f"  Gini: {m['gini']:.3f}")
    print(f"  KS  : {m['ks']:.3f}")

    # --- 4. Rank ordering (SQL) ------------------------------------------
    print("\n" + "=" * 70)
    print("RISK DECILE RANK-ORDERING  [SQL - NTILE window function]")
    print("=" * 70)
    decile = pd.read_sql(DECILE_SQL, conn)
    print(decile.to_string(index=False))

    print("\nCumulative bad capture (gains curve)  [SQL]")
    gains = pd.read_sql(GAIN_SQL, conn)
    print(gains.to_string(index=False))

    # --- 5. Strategy (SQL) -----------------------------------------------
    print("\n" + "=" * 70)
    print("APPROVAL CUTOFF STRATEGY  [SQL]")
    print("=" * 70)
    strategy = pd.read_sql(STRATEGY_SQL, conn)
    strategy.to_sql("strategy_results", conn, if_exists="replace", index=False)
    print(strategy.to_string(index=False))

    best = pd.read_sql(BEST_CUTOFF_SQL, conn).iloc[0]
    baseline = scored["actual_bad"].mean()
    reduction = 100 * (1 - best["observed_bad_rate"] / baseline)

    print("\nRECOMMENDED STRATEGY")
    print(f"  score cutoff       : {int(best['score_cutoff'])}")
    print(f"  approval rate      : {best['approval_rate']:.1%}")
    print(f"  bad rate approved  : {best['observed_bad_rate']:.2%}")
    print(f"  portfolio baseline : {baseline:.2%}")
    print(f"  bad rate reduction : {reduction:.0f}%")

    # --- 6. Export --------------------------------------------------------
    decile.to_csv(os.path.join(args.out, "risk_decile_table.csv"), index=False)
    gains.to_csv(os.path.join(args.out, "gains_curve.csv"), index=False)
    strategy.to_csv(os.path.join(args.out, "approval_cutoff_strategy.csv"),
                    index=False)
    scored.to_csv(os.path.join(args.out, "scored_test_customers.csv"),
                  index=False)

    conn.close()
    print(f"\nOutputs and database written to: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
