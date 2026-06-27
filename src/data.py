"""Data loading and leakage-safe preparation for the visa-case classification task.

Design rule: this module performs only **row-wise, data-independent** transforms
(dropping an id, fixing a sign error, deriving an age, normalizing wage by its
stated unit, encoding the target). Anything that must be *fit* from data
(one-hot categories, scaling statistics, quantile bin edges, resampling) lives in
the modeling pipeline so it is fit inside each cross-validation fold, never on the
full dataset. That is what keeps the evaluation honest.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "EasyVisa.csv"

RAW_TARGET = "case_status"
TARGET = "target"  # 1 = Certified, 0 = Denied (denial is the minority, business-critical class)
ID_COL = "case_id"

# The dataset is a derived subset of the U.S. OFLC FY2016 permanent-labor-certification
# disclosure data (see README "Source and license"). Company age is derived against
# this documented vintage rather than a magic number.
DATA_VINTAGE_YEAR = 2016

# Wage is deliberately NOT annualized. Multiplying prevailing_wage by a per-unit
# factor (Hour*2080, Week*52, ...) bakes in an unverified full-time assumption and
# manufactures implausible magnitudes (Week-unit medians near $4M). Empirically it
# is worse than useless: raw prevailing_wage has univariate AUC ~0.55 toward the
# target, the annualized version ~0.43 (below chance), because the raw magnitude
# partly proxies unit_of_wage, which is itself predictive. We keep raw
# prevailing_wage as the numeric feature and let unit_of_wage ride alongside as a
# categorical; tree models handle the within-unit scale natively.

# Feature groups consumed by the preprocessing pipeline (see preprocessing.py).
NUMERIC_FEATURES = ["no_of_employees", "company_age", "prevailing_wage"]
ORDINAL_FEATURES = {
    "education_of_employee": ["High School", "Bachelor's", "Master's", "Doctorate"],
}
BINARY_FEATURES = {  # stored as N/Y -> encoded 0/1
    "has_job_experience": ["N", "Y"],
    "requires_job_training": ["N", "Y"],
    "full_time_position": ["N", "Y"],
}
NOMINAL_FEATURES = ["continent", "region_of_employment", "unit_of_wage"]


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the raw dataset."""
    return pd.read_csv(path)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and feature-engineer with row-wise, leakage-safe transforms only.

    Steps:
      * drop the non-predictive case id,
      * repair the known negative `no_of_employees` values (a data-entry sign
        error) by taking the absolute value rather than discarding rows,
      * derive `company_age` from `yr_of_estab` against the documented vintage,
      * keep `prevailing_wage` raw (see note above on why annualizing harms it),
      * encode the target (Certified=1, Denied=0).

    Returns a model-ready frame whose feature columns are exactly the union of
    NUMERIC/ORDINAL/BINARY/NOMINAL features plus the integer `target` column.
    No fitting-from-data happens here.
    """
    out = df.copy()

    if ID_COL in out.columns:
        out = out.drop(columns=[ID_COL])

    # Sign-error repair: employee counts cannot be negative.
    out["no_of_employees"] = out["no_of_employees"].abs()

    # Derived, documented age. Guard against implausible future years.
    out["company_age"] = (DATA_VINTAGE_YEAR - out["yr_of_estab"]).clip(lower=0)
    out = out.drop(columns=["yr_of_estab"])

    # Wage kept raw; unit_of_wage carries the unit signal as a categorical.

    # Target encoding.
    out[TARGET] = (out[RAW_TARGET] == "Certified").astype(int)
    out = out.drop(columns=[RAW_TARGET])

    feature_cols = (
        NUMERIC_FEATURES
        + list(ORDINAL_FEATURES)
        + list(BINARY_FEATURES)
        + NOMINAL_FEATURES
    )
    return out[feature_cols + [TARGET]]


def split_xy(frame: pd.DataFrame):
    """Split a prepared frame into (X, y)."""
    return frame.drop(columns=[TARGET]), frame[TARGET]
