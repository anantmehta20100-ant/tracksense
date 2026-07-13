"""Maintains live risk state per tracked object, per allergen_type, by
feeding real contact events (from vision/contact_detector.py) through the
GRU trained offline on synthetic data (model/train.py).

The GRU's hidden state is carried per target track_id, so a track's risk
prediction depends on everything that has happened to it so far, not just
the current contact in isolation -- the same temporal modeling the training
setup uses, applied online.
"""

import os

import torch

from config.allergens import get_allergen_type
from model.gru_model import RiskPropagationGRU, encode_categorical

DEFAULT_CHECKPOINT_PATH = os.path.join("model", "checkpoints", "gru.pt")


class RiskState:
    def __init__(self, checkpoint_path: str = DEFAULT_CHECKPOINT_PATH):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"No trained GRU checkpoint at '{checkpoint_path}'.\n"
                "Run model/train.py first (Day 6 of AGENTS.md's plan) to generate it."
            )

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.object_vocab = checkpoint["object_vocab"]
        self.allergen_vocab = checkpoint["allergen_vocab"]

        self.model = RiskPropagationGRU(
            object_vocab_size=len(self.object_vocab),
            allergen_vocab_size=len(self.allergen_vocab),
            **checkpoint["hyperparams"],
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.risk = {}  # (track_id, allergen_type) -> current risk score
        self._hidden_states = {}  # track_id -> GRU hidden state
        self._depth_state = {}  # track_id -> current propagation depth
        self._last_contact_time = {}  # track_id -> last contact timestamp

    def get_risk(self, track_id: int, allergen_type: str) -> float:
        return self.risk.get((track_id, allergen_type), 0.0)

    def _infer_allergen_type(self, track_id: int):
        """Falls back to whichever allergen_type a track currently carries
        the most risk for, when contact_detector couldn't attribute one
        directly (see contact_detector.py: allergen_type is only looked up
        for source classes, else None)."""
        carried = {
            allergen_type: risk
            for (tid, allergen_type), risk in self.risk.items()
            if tid == track_id and risk > 0
        }
        if not carried:
            return None
        return max(carried, key=carried.get)

    def update(self, contact_event: dict, trained_model: RiskPropagationGRU = None):
        """Runs one contact event through the GRU and updates risk state for
        the target track. `trained_model` defaults to self.model; the
        parameter exists to match the interface described in AGENTS.md."""
        model = trained_model or self.model

        source_track_id = contact_event["source_track_id"]
        target_track_id = contact_event["target_track_id"]
        source_class = contact_event["source_class"]
        target_class = contact_event["target_class"]
        timestamp = contact_event["timestamp"]

        allergen_type = contact_event["allergen_type"]
        if allergen_type is None:
            allergen_type = self._infer_allergen_type(source_track_id)
            if allergen_type is None:
                return None  # no allergen risk to propagate for this pair

        if get_allergen_type(source_class) is not None:
            source_risk = 1.0  # touching the raw allergen source is always fully contaminating
        else:
            source_risk = self.get_risk(source_track_id, allergen_type)

        previous_time = self._last_contact_time.get(target_track_id)
        time_since_previous_contact = 0.0 if previous_time is None else max(0.0, timestamp - previous_time)
        self._last_contact_time[target_track_id] = timestamp

        propagation_depth = self._depth_state.get(source_track_id, 0) + 1

        # TODO(Day 7+): the live pipeline has no cleaning-event detector yet --
        # none of the 10 guaranteed object classes represents a sponge/cloth/
        # sink, and AGENTS.md doesn't specify a detection rule for "this
        # object was just cleaned". Always passing False means live risk can
        # decay via propagation depth but never reset early via cleaning,
        # unlike the synthetic training data. Revisit if a cleaning-related
        # object class gets added.
        cleaning_event = False

        event = {
            "source_object_idx": torch.tensor([[encode_categorical(self.object_vocab, source_class)]]),
            "target_object_idx": torch.tensor([[encode_categorical(self.object_vocab, target_class)]]),
            "allergen_idx": torch.tensor([[encode_categorical(self.allergen_vocab, allergen_type)]]),
            "source_risk": torch.tensor([[source_risk]]),
            "propagation_depth": torch.tensor([[float(propagation_depth)]]),
            "time_since_previous_contact": torch.tensor([[time_since_previous_contact]]),
            "cleaning_event": torch.tensor([[float(cleaning_event)]]),
        }

        hidden = self._hidden_states.get(target_track_id)
        risk_score, new_hidden = model.step(event, hidden)

        self._hidden_states[target_track_id] = new_hidden
        self._depth_state[target_track_id] = propagation_depth
        self.risk[(target_track_id, allergen_type)] = risk_score

        return risk_score
