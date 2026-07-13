"""Runs the trained GRU and all three baselines on a held-out synthetic test
set. Prints precision/recall/F1/confusion matrix per model, plus a
ranking-based Precision@K comparison of the GRU against the non-temporal
baseline -- the core research claim from AGENTS.md: does modeling the full
interaction history actually beat a model that only sees the current event?
"""

import os

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from model.gru_model import RiskPropagationGRU
from model.train import (
    CHECKPOINT_PATH,
    EVAL_RISK_THRESHOLD,
    evaluate_baseline,
    get_baselines,
    gru_predictions_and_labels,
    load_data,
    split_by_sequence,
)

RANKING_K_VALUES = (10, 50, 100)
NON_TEMPORAL_BASELINE_NAME = "Non-temporal ML (RF)"


def load_trained_gru(checkpoint_path: str = CHECKPOINT_PATH):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No trained GRU checkpoint at '{checkpoint_path}'. Run model/train.py first."
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = RiskPropagationGRU(
        object_vocab_size=len(checkpoint["object_vocab"]),
        allergen_vocab_size=len(checkpoint["allergen_vocab"]),
        **checkpoint["hyperparams"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["object_vocab"], checkpoint["allergen_vocab"]


def classification_report(name: str, y_true_continuous, y_pred_continuous, threshold: float = EVAL_RISK_THRESHOLD):
    y_true = (y_true_continuous >= threshold).astype(int)
    y_pred = (y_pred_continuous >= threshold).astype(int)
    return {
        "model": name,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def precision_at_k(y_true_continuous, y_pred_continuous, k: int, threshold: float = EVAL_RISK_THRESHOLD) -> float:
    """Ranks events by predicted risk (descending) and reports what fraction
    of the top-k predictions are truly exposed objects (y_true >= threshold).
    A model with real signal about propagation should rank truly-exposed
    objects near the top even when it didn't touch the source directly."""
    if k <= 0:
        return 0.0
    top_k_indices = np.argsort(-y_pred_continuous)[:k]
    y_true_binary = (y_true_continuous >= threshold).astype(int)
    return float(y_true_binary[top_k_indices].mean())


if __name__ == "__main__":
    df = load_data()
    train_df, test_df = split_by_sequence(df)

    gru_model, object_vocab, allergen_vocab = load_trained_gru()
    gru_preds, gru_labels = gru_predictions_and_labels(gru_model, test_df, object_vocab, allergen_vocab)

    reports = [classification_report("GRU (proposed)", gru_labels, gru_preds)]
    predictions_by_model = {"GRU (proposed)": (gru_preds, gru_labels)}

    for name, baseline in get_baselines().items():
        _, preds, labels = evaluate_baseline(name, baseline, train_df, test_df)
        reports.append(classification_report(name, labels, preds))
        predictions_by_model[name] = (preds, labels)

    print("Per-model precision / recall / F1 / confusion matrix on held-out synthetic test set:\n")
    for report in reports:
        print(f"{report['model']}:")
        print(f"  precision={report['precision']:.3f}  recall={report['recall']:.3f}  f1={report['f1']:.3f}")
        print(f"  confusion matrix [[TN, FP], [FN, TP]] = {report['confusion_matrix']}\n")

    non_temporal_preds, non_temporal_labels = predictions_by_model[NON_TEMPORAL_BASELINE_NAME]

    print(f"Ranking comparison (Precision@K): GRU vs {NON_TEMPORAL_BASELINE_NAME}")
    for k in RANKING_K_VALUES:
        k = min(k, len(test_df))
        gru_p_at_k = precision_at_k(gru_labels, gru_preds, k)
        non_temporal_p_at_k = precision_at_k(non_temporal_labels, non_temporal_preds, k)
        print(f"  P@{k}: GRU={gru_p_at_k:.3f}  {NON_TEMPORAL_BASELINE_NAME}={non_temporal_p_at_k:.3f}")
