"""Three baselines the GRU is evaluated against (AGENTS.md "Baselines to
compare against"). All expose the same .fit(train_data) / .predict(test_data)
interface, where train_data/test_data are pandas DataFrames with the
sequences.csv schema, and .predict returns a numpy array of predicted risk
scores in [0, 1] aligned row-for-row with the input.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from config.allergens import CLEANING_RISK_RESET_VALUE, RULE_BASED_DECAY_RATE
from model.gru_model import build_vocab, encode_categorical

# Risk labels are continuous; the non-temporal classifier needs a binary
# target, so labels >= this are treated as "risky" during .fit.
NON_TEMPORAL_RISK_CLASS_THRESHOLD = 0.5


class DirectContactOnlyBaseline:
    """Only flags objects that directly touched the allergen source, i.e.
    contact_type == "direct_source". Every other event (chain, unrelated,
    cleaning) is predicted as zero risk -- this baseline has no notion of
    propagation at all."""

    def fit(self, train_data):
        pass  # stateless rule, nothing to learn

    def predict(self, test_data):
        return np.where(test_data["contact_type"] == "direct_source", 1.0, 0.0)


class RuleBasedDecayBaseline:
    """Risk decreases by a fixed percentage at each hop (e.g. 100% -> 70% ->
    49% -> 34%), applied to the current source_risk. Cleaning events reset
    risk to the same near-zero value used in the synthetic label rule."""

    def __init__(self, decay_rate: float = RULE_BASED_DECAY_RATE):
        self.decay_rate = decay_rate

    def fit(self, train_data):
        pass  # fixed rule, nothing to learn

    def predict(self, test_data):
        decayed = test_data["source_risk"].to_numpy() * self.decay_rate
        return np.where(test_data["cleaning_event"].to_numpy(), CLEANING_RISK_RESET_VALUE, decayed)


class NonTemporalMLBaseline:
    """Random-forest classifier using only the current event's features --
    no sequence history, no hidden state. Demonstrates what's achievable
    without the temporal propagation modeling the GRU provides."""

    def __init__(
        self,
        risk_class_threshold: float = NON_TEMPORAL_RISK_CLASS_THRESHOLD,
        model=None,
    ):
        self.risk_class_threshold = risk_class_threshold
        self.model = model or RandomForestClassifier(n_estimators=200, random_state=42)
        self.object_vocab = None
        self.contact_type_vocab = None

    def _encode_features(self, data):
        source_idx = np.array([encode_categorical(self.object_vocab, v) for v in data["source_object"]])
        target_idx = np.array([encode_categorical(self.object_vocab, v) for v in data["target_object"]])
        contact_type_idx = np.array(
            [encode_categorical(self.contact_type_vocab, v) for v in data["contact_type"]]
        )

        return np.column_stack(
            [
                source_idx,
                target_idx,
                contact_type_idx,
                data["source_risk"].to_numpy(),
                data["propagation_depth"].to_numpy(),
                data["time_since_previous_contact"].to_numpy(),
                data["cleaning_event"].to_numpy().astype(float),
            ]
        )

    def fit(self, train_data):
        self.object_vocab = build_vocab(
            list(train_data["source_object"]) + list(train_data["target_object"])
        )
        self.contact_type_vocab = build_vocab(train_data["contact_type"])

        X = self._encode_features(train_data)
        y = (train_data["target_risk_label"] >= self.risk_class_threshold).astype(int)
        self.model.fit(X, y)

    def predict(self, test_data):
        X = self._encode_features(test_data)
        return self.model.predict_proba(X)[:, 1]
