"""Train the Random Forest cross-contact risk classifier.

Pipeline (Steps 5-6 of the build spec):
  1. Load data/risk_model/risk_events.csv and validate it against the feature
     schema (ml/risk_features.py -- the single source of feature order).
  2. Split by `scenario_id` (70/15/15, seed 42) so no contact chain leaks across
     train/val/test. Zero-overlap is asserted.
  3. Build a reproducible sklearn Pipeline: one-hot encode the two object columns
     (fixed vocabulary, handle_unknown="ignore"), pass the numeric features
     through, feed a RandomForestClassifier(class_weight="balanced").
  4. Run a small grouped-CV (GroupKFold) hyperparameter search on the training
     scenarios, scored by macro-F1.
  5. Refit the best config, report validation metrics, and save:
        model/risk_random_forest.joblib
        model/risk_model_metadata.json

Everything is downstream of, and independent from, the YOLO detector and the
GRU risk stack -- nothing here touches those artifacts.

Run: python ml/train_random_forest.py
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.risk_features import (  # noqa: E402
    CATEGORICAL_FEATURES,
    FEATURE_ORDER,
    NUMERIC_FEATURES,
    OBJECT_FEATURE_CATEGORIES,
    RISK_CLASS_LABELS,
    RISK_CLASS_TO_ID,
)

DATA_PATH = os.path.join("data", "risk_model", "risk_events.csv")
MODEL_PATH = os.path.join("model", "risk_random_forest.joblib")
METADATA_PATH = os.path.join("model", "risk_model_metadata.json")

RANDOM_SEED = 42
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)
LABEL_COLUMN = "risk_class_id"
GROUP_COLUMN = "scenario_id"

MODEL_VERSION = "risk-rf-1.0.0"


def load_events(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No risk dataset at '{path}'. Run ml/generate_risk_training_data.py first."
        )
    return pd.read_csv(path)


def validate_dataframe(df: pd.DataFrame) -> None:
    """Fail loudly if the dataset is missing schema columns or the label/group
    keys, before any training happens."""
    missing = [c for c in FEATURE_ORDER + [LABEL_COLUMN, GROUP_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"dataset is missing required columns: {missing}")
    bad_labels = set(df[LABEL_COLUMN].unique()) - set(RISK_CLASS_TO_ID.values())
    if bad_labels:
        raise ValueError(f"unexpected label ids in dataset: {bad_labels}")
    # Guard against accidental feature/latent leakage: no feature may be a latent.
    leaked = [c for c in FEATURE_ORDER if c.startswith("latent_")]
    if leaked:
        raise ValueError(f"latent columns present in FEATURE_ORDER: {leaked}")


def split_by_scenario(df: pd.DataFrame, seed: int = RANDOM_SEED, fractions=SPLIT_FRACTIONS):
    """Group-wise train/val/test split by scenario_id. Returns
    (train_df, val_df, test_df, id_sets). Asserts zero scenario overlap."""
    scenario_ids = sorted(df[GROUP_COLUMN].unique())
    random.Random(seed).shuffle(scenario_ids)

    n = len(scenario_ids)
    n_train = int(n * fractions[0])
    n_val = int(n * fractions[1])
    train_ids = set(scenario_ids[:n_train])
    val_ids = set(scenario_ids[n_train:n_train + n_val])
    test_ids = set(scenario_ids[n_train + n_val:])

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert train_ids | val_ids | test_ids == set(scenario_ids)

    def subset(ids):
        return df[df[GROUP_COLUMN].isin(ids)].reset_index(drop=True)

    return subset(train_ids), subset(val_ids), subset(test_ids), (train_ids, val_ids, test_ids)


def build_pipeline(hyperparams: dict, seed: int = RANDOM_SEED) -> Pipeline:
    """One-hot(object cols) + passthrough(numeric) -> balanced RandomForest.
    Categories are fixed from the schema so encoding is identical at inference.
    """
    encoder = OneHotEncoder(
        categories=[OBJECT_FEATURE_CATEGORIES, OBJECT_FEATURE_CATEGORIES],
        handle_unknown="ignore",
        sparse_output=False,
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("object_ohe", encoder, CATEGORICAL_FEATURES),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ],
        remainder="drop",
    )
    classifier = RandomForestClassifier(
        random_state=seed,
        class_weight="balanced",
        n_jobs=-1,
        **hyperparams,
    )
    return Pipeline([("preprocess", preprocessor), ("random_forest", classifier)])


def _feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Select exactly the schema features, in canonical order. This is the ONLY
    place features are pulled from the dataframe -- latent/label columns present
    in the CSV are never selected."""
    return df[FEATURE_ORDER].copy()


def run_search(train_df: pd.DataFrame, base_hyperparams: dict, seed: int, n_splits: int = 3):
    """Small grouped-CV hyperparameter search on the training scenarios.
    Returns (best_pipeline, best_params, cv_summary)."""
    X = _feature_matrix(train_df)
    y = train_df[LABEL_COLUMN].to_numpy()
    groups = train_df[GROUP_COLUMN].to_numpy()

    param_grid = {
        "random_forest__n_estimators": sorted({base_hyperparams["n_estimators"], 300}),
        "random_forest__max_depth": [None, 16],
        "random_forest__min_samples_leaf": sorted({base_hyperparams["min_samples_leaf"], 1, 3}),
        "random_forest__max_features": ["sqrt", 0.5],
    }
    pipeline = build_pipeline(
        {k: base_hyperparams[k] for k in ("n_estimators", "max_depth", "min_samples_leaf", "max_features")},
        seed=seed,
    )
    search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=GroupKFold(n_splits=n_splits),
        n_jobs=1,
        refit=True,
    )
    search.fit(X, y, groups=groups)

    best_params = {k.replace("random_forest__", ""): v for k, v in search.best_params_.items()}
    cv_summary = {
        "scoring": "f1_macro",
        "n_splits": n_splits,
        "best_cv_score": float(search.best_score_),
        "n_candidates": len(search.cv_results_["params"]),
    }
    return search.best_estimator_, best_params, cv_summary


def _resolve_max_depth(value):
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in ("none", "", "-1"):
        return None
    return int(value)


def train(args) -> dict:
    df = load_events(args.data)
    validate_dataframe(df)
    train_df, val_df, test_df, (train_ids, val_ids, test_ids) = split_by_scenario(
        df, seed=args.seed, fractions=SPLIT_FRACTIONS
    )

    print(f"Loaded {len(df)} events / {df[GROUP_COLUMN].nunique()} scenarios.")
    print(f"Split (by scenario_id, seed {args.seed}): "
          f"train {len(train_ids)} scen / {len(train_df)} rows, "
          f"val {len(val_ids)} / {len(val_df)}, test {len(test_ids)} / {len(test_df)}.")
    print("Zero scenario overlap between splits: CONFIRMED")

    base_hyperparams = {
        "n_estimators": args.n_estimators,
        "max_depth": _resolve_max_depth(args.max_depth),
        "min_samples_leaf": args.min_samples_leaf,
        "max_features": args.max_features,
    }

    if args.search:
        print("Running grouped-CV hyperparameter search (GroupKFold, macro-F1)...")
        model, chosen_hyperparams, cv_summary = run_search(train_df, base_hyperparams, args.seed)
        print(f"  best CV macro-F1 = {cv_summary['best_cv_score']:.3f} "
              f"over {cv_summary['n_candidates']} candidates")
    else:
        print("Training with fixed hyperparameters (no search).")
        model = build_pipeline(base_hyperparams, seed=args.seed)
        model.fit(_feature_matrix(train_df), train_df[LABEL_COLUMN].to_numpy())
        chosen_hyperparams = base_hyperparams
        cv_summary = {"scoring": "f1_macro", "n_splits": 0, "best_cv_score": None, "n_candidates": 0}

    print(f"Chosen hyperparameters: {chosen_hyperparams}")

    # Validation-set report (held-out scenarios, not used in the search folds'
    # test partitions beyond CV -- an independent sanity check).
    val_pred = model.predict(_feature_matrix(val_df))
    val_true = val_df[LABEL_COLUMN].to_numpy()
    val_macro_f1 = f1_score(val_true, val_pred, average="macro")
    val_bal_acc = balanced_accuracy_score(val_true, val_pred)
    print(f"Validation: macro-F1 = {val_macro_f1:.3f}, balanced accuracy = {val_bal_acc:.3f}")

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    joblib.dump(model, args.model_out)

    class_dist = train_df["risk_class"].value_counts().reindex(RISK_CLASS_LABELS).fillna(0).astype(int)
    metadata = {
        "model_version": MODEL_VERSION,
        "model_type": "RandomForestClassifier",
        "task": "relative cross-contact risk (LOW/MEDIUM/HIGH) -- synthetic development model",
        "feature_order": FEATURE_ORDER,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "object_categories": OBJECT_FEATURE_CATEGORIES,
        "class_labels": RISK_CLASS_LABELS,
        "class_label_ids": RISK_CLASS_TO_ID,
        "random_seed": args.seed,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_path": args.data,
        "dataset_row_count": int(len(df)),
        "n_scenarios": int(df[GROUP_COLUMN].nunique()),
        "split_fractions": {"train": SPLIT_FRACTIONS[0], "val": SPLIT_FRACTIONS[1], "test": SPLIT_FRACTIONS[2]},
        "n_train_rows": int(len(train_df)),
        "n_val_rows": int(len(val_df)),
        "n_test_rows": int(len(test_df)),
        "n_train_scenarios": int(len(train_ids)),
        "n_val_scenarios": int(len(val_ids)),
        "n_test_scenarios": int(len(test_ids)),
        "hyperparameters": chosen_hyperparams,
        "cv": cv_summary,
        "validation_macro_f1": float(val_macro_f1),
        "validation_balanced_accuracy": float(val_bal_acc),
        "train_class_distribution": {label: int(class_dist[label]) for label in RISK_CLASS_LABELS},
        "data_disclaimer": (
            "Trained on SYNTHETIC development data. Outputs are relative "
            "cross-contact risk, not measured allergen concentration."
        ),
    }
    with open(args.metadata_out, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Saved model    -> {args.model_out}")
    print(f"Saved metadata -> {args.metadata_out}")
    return metadata


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the Random Forest risk classifier.")
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--model-out", default=MODEL_PATH)
    parser.add_argument("--metadata-out", default=METADATA_PATH)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", default="none", help="int or 'none'")
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--max-features", default="sqrt")
    search = parser.add_mutually_exclusive_group()
    search.add_argument("--search", dest="search", action="store_true", help="grouped-CV search (default)")
    search.add_argument("--no-search", dest="search", action="store_false")
    parser.set_defaults(search=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
