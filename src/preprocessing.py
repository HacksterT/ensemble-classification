"""Preprocessing pipeline.

All learned transforms (one-hot categories, scaling statistics) are assembled
here as a ColumnTransformer so they are fit *inside* each CV fold by the
surrounding pipeline, not once on the full dataset. This is the leakage guard the
repo's thesis depends on.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .data import (
    BINARY_FEATURES,
    NOMINAL_FEATURES,
    NUMERIC_FEATURES,
    ORDINAL_FEATURES,
)


def build_preprocessor(scale_numeric: bool = False) -> ColumnTransformer:
    """Construct the feature preprocessing transformer.

    Parameters
    ----------
    scale_numeric:
        Standardize numeric columns. Off for tree models (which are
        scale-invariant), on for the linear baseline.
    """
    ordinal_cols = list(ORDINAL_FEATURES)
    ordinal_categories = [ORDINAL_FEATURES[c] for c in ordinal_cols]

    binary_cols = list(BINARY_FEATURES)
    binary_categories = [BINARY_FEATURES[c] for c in binary_cols]

    numeric_transformer = StandardScaler() if scale_numeric else "passthrough"

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            (
                "ordinal",
                OrdinalEncoder(
                    categories=ordinal_categories,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                ordinal_cols,
            ),
            (
                "binary",
                OrdinalEncoder(
                    categories=binary_categories,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                binary_cols,
            ),
            (
                "nominal",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                NOMINAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
