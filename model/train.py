"""Loads the synthetic contact-event dataset, trains the GRU and all three
baselines, evaluates all four on a held-out split, and saves the trained GRU.

Held-out split is done by sequence_id (not by row) so events from the same
kitchen scene never leak across train/test.
"""

import os
import random

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

from model.baselines import DirectContactOnlyBaseline, NonTemporalMLBaseline, RuleBasedDecayBaseline
from model.gru_model import (
    DEFAULT_ALLERGEN_EMBED_DIM,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_OBJECT_EMBED_DIM,
    RiskPropagationGRU,
    build_vocab,
    encode_categorical,
)

DATA_PATH = os.path.join("data", "synthetic", "sequences.csv")
CHECKPOINT_PATH = os.path.join("model", "checkpoints", "gru.pt")

TEST_FRACTION = 0.2
RANDOM_SEED = 42

NUM_EPOCHS = 15
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
NUM_GRU_LAYERS = 1

# Continuous risk scores/labels are turned into low/high classes at this
# threshold for accuracy/F1 reporting, shared across all four models so the
# comparison is apples-to-apples.
EVAL_RISK_THRESHOLD = 0.5

GRU_HYPERPARAMS = {
    "object_embed_dim": DEFAULT_OBJECT_EMBED_DIM,
    "allergen_embed_dim": DEFAULT_ALLERGEN_EMBED_DIM,
    "hidden_size": DEFAULT_HIDDEN_SIZE,
    "num_gru_layers": NUM_GRU_LAYERS,
}


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def split_by_sequence(df: pd.DataFrame, test_fraction: float = TEST_FRACTION, seed: int = RANDOM_SEED):
    sequence_ids = list(df["sequence_id"].unique())
    random.Random(seed).shuffle(sequence_ids)

    split_point = int(len(sequence_ids) * (1 - test_fraction))
    train_ids = set(sequence_ids[:split_point])
    test_ids = set(sequence_ids[split_point:])

    train_df = df[df["sequence_id"].isin(train_ids)].reset_index(drop=True)
    test_df = df[df["sequence_id"].isin(test_ids)].reset_index(drop=True)
    return train_df, test_df


def build_gru_vocabs(train_df: pd.DataFrame):
    object_vocab = build_vocab(list(train_df["source_object"]) + list(train_df["target_object"]))
    allergen_vocab = build_vocab(train_df["allergen_type"])
    return object_vocab, allergen_vocab


def build_padded_batch(df: pd.DataFrame, sequence_ids, object_vocab: dict, allergen_vocab: dict):
    """Builds a padded (batch_size, max_len) batch for a list of sequence_ids.

    Returns (batch: dict of feature tensors, labels: tensor, mask: tensor of
    1s for real positions / 0s for padding).
    """
    sequences = [
        df[df["sequence_id"] == sid].sort_values("event_index") for sid in sequence_ids
    ]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)

    source_idx = torch.zeros(batch_size, max_len, dtype=torch.long)
    target_idx = torch.zeros(batch_size, max_len, dtype=torch.long)
    allergen_idx = torch.zeros(batch_size, max_len, dtype=torch.long)
    source_risk = torch.zeros(batch_size, max_len)
    propagation_depth = torch.zeros(batch_size, max_len)
    time_since_previous_contact = torch.zeros(batch_size, max_len)
    cleaning_event = torch.zeros(batch_size, max_len)
    labels = torch.zeros(batch_size, max_len)
    mask = torch.zeros(batch_size, max_len)

    for i, seq in enumerate(sequences):
        length = len(seq)
        source_idx[i, :length] = torch.tensor(
            [encode_categorical(object_vocab, v) for v in seq["source_object"]]
        )
        target_idx[i, :length] = torch.tensor(
            [encode_categorical(object_vocab, v) for v in seq["target_object"]]
        )
        allergen_idx[i, :length] = torch.tensor(
            [encode_categorical(allergen_vocab, v) for v in seq["allergen_type"]]
        )
        source_risk[i, :length] = torch.tensor(seq["source_risk"].to_numpy(), dtype=torch.float32)
        propagation_depth[i, :length] = torch.tensor(
            seq["propagation_depth"].to_numpy(), dtype=torch.float32
        )
        time_since_previous_contact[i, :length] = torch.tensor(
            seq["time_since_previous_contact"].to_numpy(), dtype=torch.float32
        )
        cleaning_event[i, :length] = torch.tensor(
            seq["cleaning_event"].to_numpy().astype(float), dtype=torch.float32
        )
        labels[i, :length] = torch.tensor(seq["target_risk_label"].to_numpy(), dtype=torch.float32)
        mask[i, :length] = 1.0

    batch = {
        "source_object_idx": source_idx,
        "target_object_idx": target_idx,
        "allergen_idx": allergen_idx,
        "source_risk": source_risk,
        "propagation_depth": propagation_depth,
        "time_since_previous_contact": time_since_previous_contact,
        "cleaning_event": cleaning_event,
    }
    return batch, labels, mask


def train_gru(
    train_df: pd.DataFrame,
    object_vocab: dict,
    allergen_vocab: dict,
    epochs: int = NUM_EPOCHS,
    lr: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
    seed: int = RANDOM_SEED,
) -> RiskPropagationGRU:
    torch.manual_seed(seed)

    model = RiskPropagationGRU(
        object_vocab_size=len(object_vocab),
        allergen_vocab_size=len(allergen_vocab),
        **GRU_HYPERPARAMS,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss(reduction="none")

    sequence_ids = list(train_df["sequence_id"].unique())

    for epoch in range(epochs):
        random.shuffle(sequence_ids)
        epoch_loss, num_batches = 0.0, 0

        for start in range(0, len(sequence_ids), batch_size):
            batch_ids = sequence_ids[start : start + batch_size]
            batch, labels, mask = build_padded_batch(train_df, batch_ids, object_vocab, allergen_vocab)

            predictions, _ = model(batch)
            predictions = predictions.squeeze(-1)
            loss = (loss_fn(predictions, labels) * mask).sum() / mask.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        print(f"  epoch {epoch + 1}/{epochs} - loss: {epoch_loss / num_batches:.4f}")

    return model


def gru_predictions_and_labels(
    model: RiskPropagationGRU,
    df: pd.DataFrame,
    object_vocab: dict,
    allergen_vocab: dict,
    batch_size: int = BATCH_SIZE,
):
    model.eval()
    sequence_ids = list(df["sequence_id"].unique())
    all_preds, all_labels = [], []

    with torch.no_grad():
        for start in range(0, len(sequence_ids), batch_size):
            batch_ids = sequence_ids[start : start + batch_size]
            batch, labels, mask = build_padded_batch(df, batch_ids, object_vocab, allergen_vocab)
            predictions, _ = model(batch)
            predictions = predictions.squeeze(-1)

            valid = mask.bool()
            all_preds.append(predictions[valid])
            all_labels.append(labels[valid])

    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


def binary_metrics(name: str, y_true_continuous, y_pred_continuous, threshold: float = EVAL_RISK_THRESHOLD):
    y_true = (y_true_continuous >= threshold).astype(int)
    y_pred = (y_pred_continuous >= threshold).astype(int)
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def evaluate_baseline(name: str, baseline, train_df: pd.DataFrame, test_df: pd.DataFrame):
    baseline.fit(train_df)
    predictions = baseline.predict(test_df)
    labels = test_df["target_risk_label"].to_numpy()
    return binary_metrics(name, labels, predictions), predictions, labels


def get_baselines():
    return {
        "Direct-contact-only": DirectContactOnlyBaseline(),
        "Rule-based decay": RuleBasedDecayBaseline(),
        "Non-temporal ML (RF)": NonTemporalMLBaseline(),
    }


if __name__ == "__main__":
    df = load_data()
    train_df, test_df = split_by_sequence(df)
    object_vocab, allergen_vocab = build_gru_vocabs(train_df)

    print(f"Loaded {len(df)} events, {df['sequence_id'].nunique()} sequences "
          f"({len(train_df)} train / {len(test_df)} test rows).")

    print("Training GRU...")
    gru_model = train_gru(train_df, object_vocab, allergen_vocab)
    gru_preds, gru_labels = gru_predictions_and_labels(gru_model, test_df, object_vocab, allergen_vocab)

    results = [binary_metrics("GRU (proposed)", gru_labels, gru_preds)]

    print("Training baselines...")
    for name, baseline in get_baselines().items():
        metrics, _, _ = evaluate_baseline(name, baseline, train_df, test_df)
        results.append(metrics)

    print("\nComparison on held-out synthetic test set:")
    print(pd.DataFrame(results).to_string(index=False))

    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    torch.save(
        {
            "model_state_dict": gru_model.state_dict(),
            "object_vocab": object_vocab,
            "allergen_vocab": allergen_vocab,
            "hyperparams": GRU_HYPERPARAMS,
        },
        CHECKPOINT_PATH,
    )
    print(f"\nSaved trained GRU to {CHECKPOINT_PATH}")
