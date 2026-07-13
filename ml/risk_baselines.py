"""Baselines the Random Forest risk classifier is compared against (Step 7).

The Random Forest must earn its place: it should beat a trivial majority-class
predictor and a simple hand-written rule, and be compared fairly against a
linear model (logistic regression) on the SAME held-out test scenarios.

All baselines share one interface so evaluate/evaluate_random_forest.py can
treat them (and the RF pipeline) uniformly:

    baseline.fit(train_df)            # train_df: risk_events.csv schema
    baseline.predict(df)  -> np.ndarray[int]     class ids (0/1/2)
    baseline.predict_proba(df) -> np.ndarray     shape (n, 3), columns = class ids

Predictions/probabilities are aligned to ml.risk_features.RISK_CLASS_LABELS
order (LOW=0, MEDIUM=1, HIGH=2).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.risk_features import (  # noqa: E402
    CATEGORICAL_FEATURES,
    FEATURE_ORDER,
    NUMERIC_FEATURES,
    OBJECT_FEATURE_CATEGORIES,
    RISK_CLASS_LABELS,
    RISK_CLASS_TO_ID,
)

LABEL_COLUMN = "risk_class_id"
N_CLASSES = len(RISK_CLASS_LABELS)


def _feature_matrix(df):
    return df[FEATURE_ORDER].copy()


class MajorityClassBaseline:
    """Always predicts the most frequent class in the training data.
    predict_proba returns the training class frequencies for every row."""

    def __init__(self):
        self.majority_class = 0
        self.class_frequencies = np.zeros(N_CLASSES)

    def fit(self, train_df):
        counts = np.zeros(N_CLASSES)
        for class_id, count in train_df[LABEL_COLUMN].value_counts().items():
            counts[int(class_id)] = count
        self.class_frequencies = counts / counts.sum()
        self.majority_class = int(np.argmax(counts))
        return self

    def predict(self, df):
        return np.full(len(df), self.majority_class, dtype=int)

    def predict_proba(self, df):
        return np.tile(self.class_frequencies, (len(df), 1))


class DirectContactRuleBaseline:
    """Simple, interpretable rule -- no learning. Embodies "flag direct allergen
    contact, trust cleaning, otherwise scale with the source's current risk":

        cleaning_detected              -> LOW
        direct allergen source contact -> HIGH
        source_current_risk >= 0.5     -> MEDIUM
        source_current_risk >= 0.2     -> MEDIUM
        otherwise                      -> LOW
    """

    HIGH = RISK_CLASS_TO_ID["HIGH"]
    MEDIUM = RISK_CLASS_TO_ID["MEDIUM"]
    LOW = RISK_CLASS_TO_ID["LOW"]

    def fit(self, train_df):
        return self  # stateless

    def predict(self, df):
        preds = np.full(len(df), self.LOW, dtype=int)
        cleaning = df["cleaning_detected"].to_numpy().astype(int)
        is_allergen = df["is_source_allergen"].to_numpy().astype(int)
        source_risk = df["source_current_risk"].to_numpy().astype(float)

        preds = np.where(source_risk >= 0.2, self.MEDIUM, preds)
        preds = np.where(source_risk >= 0.5, self.MEDIUM, preds)
        preds = np.where(is_allergen == 1, self.HIGH, preds)
        preds = np.where(cleaning == 1, self.LOW, preds)
        return preds

    def predict_proba(self, df):
        preds = self.predict(df)
        proba = np.zeros((len(df), N_CLASSES))
        proba[np.arange(len(df)), preds] = 1.0
        return proba


class LogisticRegressionBaseline:
    """Multinomial logistic regression on the same features (one-hot objects +
    standardized numerics). A fair linear comparison point for the RF."""

    def __init__(self, seed: int = 42, max_iter: int = 2000):
        encoder = OneHotEncoder(
            categories=[OBJECT_FEATURE_CATEGORIES, OBJECT_FEATURE_CATEGORIES],
            handle_unknown="ignore",
            sparse_output=False,
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("object_ohe", encoder, CATEGORICAL_FEATURES),
                ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ],
            remainder="drop",
        )
        self.pipeline = Pipeline([
            ("preprocess", preprocessor),
            ("logreg", LogisticRegression(
                class_weight="balanced",
                max_iter=max_iter,
                random_state=seed,
            )),
        ])

    def fit(self, train_df):
        self.pipeline.fit(_feature_matrix(train_df), train_df[LABEL_COLUMN].to_numpy())
        return self

    def predict(self, df):
        return self.pipeline.predict(_feature_matrix(df)).astype(int)

    def predict_proba(self, df):
        proba = self.pipeline.predict_proba(_feature_matrix(df))
        # Reorder columns to canonical class-id order in case classes_ differs.
        classes = list(self.pipeline.named_steps["logreg"].classes_)
        ordered = np.zeros((len(df), N_CLASSES))
        for col, class_id in enumerate(classes):
            ordered[:, int(class_id)] = proba[:, col]
        return ordered


def get_baselines(seed: int = 42):
    """Return the ordered dict of baseline name -> fresh instance."""
    return {
        "Majority class": MajorityClassBaseline(),
        "Direct-contact rule": DirectContactRuleBaseline(),
        "Logistic Regression": LogisticRegressionBaseline(seed=seed),
    }
