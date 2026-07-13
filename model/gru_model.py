"""GRU-based temporal risk-propagation model (see AGENTS.md "The ML problem").

Predicts a target object's risk score for a given event, conditioned on the
full history of prior events in its chain via the GRU's hidden state -- not
just the current contact in isolation.
"""

import torch
import torch.nn as nn

UNK_TOKEN = "<unk>"

# Feature-scaling constants (model-internal, not a domain/business threshold,
# so these live alongside the architecture rather than in config/allergens.py).
PROPAGATION_DEPTH_SCALE = 10.0
TIME_SINCE_PREVIOUS_SCALE = 30.0

DEFAULT_OBJECT_EMBED_DIM = 16
DEFAULT_ALLERGEN_EMBED_DIM = 4
DEFAULT_HIDDEN_SIZE = 64


def build_vocab(values):
    """Builds a {token: index} vocab from an iterable of strings, reserving
    index 0 for unknown/unseen tokens encountered later at inference time."""
    unique_values = sorted(set(values))
    vocab = {UNK_TOKEN: 0}
    for value in unique_values:
        if value not in vocab:
            vocab[value] = len(vocab)
    return vocab


def encode_categorical(vocab: dict, value: str) -> int:
    return vocab.get(value, vocab[UNK_TOKEN])


class RiskPropagationGRU(nn.Module):
    def __init__(
        self,
        object_vocab_size: int,
        allergen_vocab_size: int,
        object_embed_dim: int = DEFAULT_OBJECT_EMBED_DIM,
        allergen_embed_dim: int = DEFAULT_ALLERGEN_EMBED_DIM,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        num_gru_layers: int = 1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_gru_layers = num_gru_layers

        self.object_embedding = nn.Embedding(object_vocab_size, object_embed_dim)
        self.allergen_embedding = nn.Embedding(allergen_vocab_size, allergen_embed_dim)

        num_continuous_features = 4  # source_risk, propagation_depth, time_since_previous_contact, cleaning_event
        input_size = object_embed_dim * 2 + allergen_embed_dim + num_continuous_features

        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_gru_layers, batch_first=True)
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def _build_features(self, batch: dict) -> torch.Tensor:
        """batch: dict of tensors, each shaped (batch_size, seq_len), except
        the continuous fields which are also (batch_size, seq_len)."""
        source_emb = self.object_embedding(batch["source_object_idx"])
        target_emb = self.object_embedding(batch["target_object_idx"])
        allergen_emb = self.allergen_embedding(batch["allergen_idx"])

        continuous = torch.stack(
            [
                batch["source_risk"],
                batch["propagation_depth"] / PROPAGATION_DEPTH_SCALE,
                batch["time_since_previous_contact"] / TIME_SINCE_PREVIOUS_SCALE,
                batch["cleaning_event"],
            ],
            dim=-1,
        )
        return torch.cat([source_emb, target_emb, allergen_emb, continuous], dim=-1)

    def forward(self, batch: dict, hidden: torch.Tensor = None):
        """Full-sequence training forward pass.

        Returns (risk: (batch_size, seq_len, 1) in [0, 1], hidden: final GRU
        hidden state).
        """
        features = self._build_features(batch)
        gru_out, hidden = self.gru(features, hidden)
        risk = torch.sigmoid(self.prediction_head(gru_out))
        return risk, hidden

    def step(self, event: dict, hidden: torch.Tensor = None):
        """Single-event online inference for the live pipeline
        (pipeline/risk_state.py). `event` holds scalar tensors shaped (1, 1).
        `hidden` is the object's previous hidden state, or None to start a
        fresh chain. Returns (risk: float, new_hidden)."""
        with torch.no_grad():
            risk, new_hidden = self.forward(event, hidden)
        return risk.item(), new_hidden

    def init_hidden(self, batch_size: int = 1) -> torch.Tensor:
        return torch.zeros(self.num_gru_layers, batch_size, self.hidden_size)
