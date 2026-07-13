"""Synthetic contact-event generator for the Random Forest risk model.

Why synthetic: there is no public dataset with true allergen-transfer labels
for kitchen contact chains (see README / AGENTS.md "Dataset strategy"). This
script fabricates realistic, *labeled* contact-event sequences for development.

Anti-triviality (Step 4 of the build spec): labels are NOT produced by one
deterministic formula like `risk = previous_risk * 0.7`. Instead each scenario
runs under a HIDDEN transfer regime and several HIDDEN latent variables that are
NEVER exposed as model features:

    - regime            (low/medium/high transfer, dry contact, sticky spread)
    - latent stickiness (per event, drawn from the regime)
    - latent cleaning efficacy (per cleaning event -> models INCOMPLETE cleaning)
    - per-scenario depth decay, time decay, observation noise

The observable features the model DOES see (contact duration, overlap, distance,
propagation depth, cleaning flag, repeat count, source/target risk, ...) only
partially determine the label, because the hidden regime + noise + label-
boundary jitter sit between the features and the class. This stops the Random
Forest from simply reconstructing the generating rule, while keeping physically
sensible *tendencies*:

    - direct allergen-source contact usually (not always) raises risk
    - higher source risk / longer / repeated contact tend to raise risk
    - greater propagation depth and elapsed time tend to lower risk
    - cleaning generally lowers risk (incompletely, if efficacy is low)
    - unrelated no-contact events stay low risk

Outputs (Step 4):
    data/risk_model/risk_events.csv       one row per contact event
    data/risk_model/scenario_metadata.csv one row per scenario (grouping key)

Both carry `scenario_id` so downstream splitting can be grouped and leakage-free.
Per-event/per-scenario latent columns are written with a `latent_` prefix purely
for transparency/analysis; training selects only ml.risk_features.FEATURE_ORDER,
so they can never enter the model.

Run: python ml/generate_risk_training_data.py --num-scenarios 5000 --seed 42
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.allergens import ALLERGEN_SOURCE_CLASSES, OBJECT_CLASSES  # noqa: E402
from ml.risk_features import (  # noqa: E402
    CLEANING_SUPPLY_LABEL,
    FEATURE_ORDER,
    RISK_ID_TO_CLASS,
    events_to_frame,
)

# ---------------------------------------------------------------------------
# Object pools. `counter` is excluded from generated events because the live
# 8-class YOLO model cannot detect it yet (ml/class_schema.py); it stays in the
# feature vocabulary so inference won't break if counter detection is added.
# ---------------------------------------------------------------------------
ALLERGEN_SOURCES = list(ALLERGEN_SOURCE_CLASSES.keys())
GENERATION_OBJECTS = [c for c in OBJECT_CLASSES if c not in ("counter", *ALLERGEN_SOURCES)]

DEFAULT_NUM_SCENARIOS = 5000
DEFAULT_SEED = 42
DEFAULT_OUTPUT_DIR = os.path.join("data", "risk_model")

# ---------------------------------------------------------------------------
# HIDDEN regime parameters (never written as features).
# ---------------------------------------------------------------------------
REGIME_COEFF = {
    "low_transfer": (0.25, 0.50),
    "medium_transfer": (0.55, 0.85),
    "high_transfer": (0.90, 1.20),
    "dry_contact": (0.12, 0.28),
    "sticky_spread": (1.05, 1.40),
}
REGIME_STICKINESS = {
    "low_transfer": (0.60, 1.00),
    "medium_transfer": (0.70, 1.05),
    "high_transfer": (0.80, 1.15),
    "dry_contact": (0.30, 0.60),
    "sticky_spread": (0.95, 1.35),
}

# Label-class boundaries applied to the (hidden) continuous target contamination.
LOW_MED_BOUNDARY = 0.30
MED_HIGH_BOUNDARY = 0.62
BOUNDARY_JITTER_STD = 0.04  # blurs class boundaries -> label noise, no perfect fit

# ---------------------------------------------------------------------------
# Scenario presets. Each controls chain shape + allowed hidden regimes.
# `weight` is the sampling probability of that scenario type.
# ---------------------------------------------------------------------------
SCENARIO_PRESETS = {
    "direct_allergen_chain": dict(
        weight=0.20, allergen_start=True, extra=(2, 6),
        actions={"chain": 0.7, "branch": 0.1, "repeat": 0.05, "clean": 0.05, "unrelated": 0.1},
        regimes=["low_transfer", "medium_transfer", "high_transfer"],
        duration="normal", clean_efficacy="high"),
    "allergen_hand_chain": dict(
        weight=0.12, allergen_start=True, extra=(1, 4), first_target=["hand", "cutlery"],
        actions={"chain": 0.75, "branch": 0.1, "repeat": 0.05, "clean": 0.0, "unrelated": 0.1},
        regimes=["medium_transfer", "high_transfer", "sticky_spread"],
        duration="normal", clean_efficacy="high"),
    "cleaning_scenario": dict(
        weight=0.12, allergen_start=True, extra=(2, 5),
        actions={"chain": 0.45, "branch": 0.05, "repeat": 0.05, "clean": 0.4, "unrelated": 0.05},
        regimes=["medium_transfer", "high_transfer", "sticky_spread"],
        duration="normal", clean_efficacy="high"),
    "incomplete_cleaning": dict(
        weight=0.10, allergen_start=True, extra=(2, 5),
        actions={"chain": 0.45, "branch": 0.05, "repeat": 0.05, "clean": 0.4, "unrelated": 0.05},
        regimes=["medium_transfer", "high_transfer", "sticky_spread"],
        duration="normal", clean_efficacy="low"),
    "unrelated_contacts": dict(
        weight=0.12, allergen_start=False, extra=(2, 5),
        actions={"unrelated": 1.0},
        regimes=["low_transfer"], duration="normal", clean_efficacy="high"),
    "repeated_contact_chain": dict(
        weight=0.10, allergen_start=True, extra=(3, 7),
        actions={"chain": 0.35, "branch": 0.05, "repeat": 0.5, "clean": 0.05, "unrelated": 0.05},
        regimes=["low_transfer", "medium_transfer", "high_transfer", "sticky_spread"],
        duration="normal", clean_efficacy="high"),
    "branching_chain": dict(
        weight=0.08, allergen_start=True, extra=(3, 7),
        actions={"chain": 0.35, "branch": 0.5, "repeat": 0.05, "clean": 0.05, "unrelated": 0.05},
        regimes=["medium_transfer", "high_transfer"],
        duration="normal", clean_efficacy="high"),
    "prolonged_contact": dict(
        weight=0.06, allergen_start=True, extra=(2, 5),
        actions={"chain": 0.7, "branch": 0.1, "repeat": 0.1, "clean": 0.0, "unrelated": 0.1},
        regimes=["medium_transfer", "high_transfer", "sticky_spread"],
        duration="long", clean_efficacy="high"),
    "short_dry_contact": dict(
        weight=0.06, allergen_start=True, extra=(2, 5),
        actions={"chain": 0.75, "branch": 0.1, "repeat": 0.05, "clean": 0.0, "unrelated": 0.1},
        regimes=["dry_contact", "low_transfer"],
        duration="short", clean_efficacy="high"),
    "deep_chain": dict(
        weight=0.04, allergen_start=True, extra=(6, 11),
        actions={"chain": 0.85, "branch": 0.05, "repeat": 0.05, "clean": 0.0, "unrelated": 0.05},
        regimes=["medium_transfer", "high_transfer"],
        duration="normal", clean_efficacy="high"),
}

DURATION_BANDS = {"short": (0.2, 2.0), "normal": (1.0, 12.0), "long": (10.0, 60.0)}
CLEAN_EFFICACY_BANDS = {"high": (0.80, 1.00), "low": (0.40, 0.70)}

TIME_GAP_MEAN_SECONDS = 5.0


# ---------------------------------------------------------------------------
# Physically-sensible (but non-deterministic) transfer factors.
# ---------------------------------------------------------------------------
def _duration_factor(duration: float) -> float:
    """Saturating gain in [0.75, 1.0]: longer contact transfers more, with
    diminishing returns."""
    return 0.75 + 0.25 * (1.0 - math.exp(-duration / 8.0))


def _overlap_factor(overlap: float, distance: float) -> float:
    """More bbox overlap and less distance -> more transfer. Can exceed 1.0 for
    a firm, fully-overlapping contact and drop toward 0.5 for a distant graze."""
    return (0.7 + 0.5 * overlap) / (1.0 + 0.2 * distance)


def _repeat_factor(repeat_count: int) -> float:
    return min(1.5, 1.0 + 0.12 * repeat_count)


def _bucket(continuous_risk: float, rng: np.random.Generator) -> int:
    jittered = continuous_risk + rng.normal(0.0, BOUNDARY_JITTER_STD)
    if jittered < LOW_MED_BOUNDARY:
        return 0
    if jittered < MED_HIGH_BOUNDARY:
        return 1
    return 2


def _draw_geometry(rng: np.random.Generator, duration_band: str):
    duration = float(rng.uniform(*DURATION_BANDS[duration_band]))
    overlap = float(rng.uniform(0.05, 0.95))
    # Distance is anti-correlated with overlap (touching objects overlap more).
    distance = float(max(0.0, (1.0 - overlap) * rng.uniform(0.4, 2.5)))
    return duration, overlap, distance


def _weighted_choice(rng: np.random.Generator, options: dict) -> str:
    keys = list(options.keys())
    probs = np.array([options[k] for k in keys], dtype=float)
    probs = probs / probs.sum()
    return keys[int(rng.choice(len(keys), p=probs))]


class _ObjectState:
    """Per-object running state within a single scenario (one instance per
    class name, matching model/synthetic_data.py's simplification)."""

    __slots__ = ("observed_risk", "latent", "depth", "last_time", "contact_count", "contaminated")

    def __init__(self):
        self.observed_risk = {}   # class -> noisy risk proxy carried downstream
        self.latent = {}          # class -> true latent contamination
        self.depth = {}           # class -> hops from allergen source
        self.last_time = {}       # class -> last contact sim time
        self.contact_count = {}   # class -> events involved in so far
        self.contaminated = []    # classes with latent > 0, in first-touch order


def generate_scenario(scenario_id: int, scenario_type: str, rng: np.random.Generator):
    """Generate one labeled contact chain. Returns (event_rows, scenario_meta)."""
    preset = SCENARIO_PRESETS[scenario_type]
    regime = preset["regimes"][int(rng.integers(len(preset["regimes"])))]
    coeff_lo, coeff_hi = REGIME_COEFF[regime]
    stick_lo, stick_hi = REGIME_STICKINESS[regime]

    depth_decay = float(rng.uniform(0.60, 0.92))
    time_decay_rate = float(rng.uniform(0.05, 0.45))
    obs_noise = float(rng.uniform(0.03, 0.06))
    label_noise = float(rng.uniform(0.03, 0.08))

    state = _ObjectState()
    events = []
    sim_time = 0.0
    event_index = 0

    def transfer_label(base_risk, is_allergen, duration, overlap, distance,
                       repeat_count, depth, seconds_since_exposure):
        coeff = float(rng.uniform(coeff_lo, coeff_hi))
        stickiness = float(rng.uniform(stick_lo, stick_hi))
        transfer = (coeff * _duration_factor(duration) * _overlap_factor(overlap, distance)
                    * stickiness * _repeat_factor(repeat_count))
        depth_factor = depth_decay ** max(0, depth)
        time_factor = math.exp(-time_decay_rate * seconds_since_exposure / 60.0)
        base = 1.0 if is_allergen else base_risk
        raw = base * transfer * depth_factor * time_factor
        raw = float(np.clip(raw + rng.normal(0.0, label_noise), 0.0, 1.0))
        return raw, coeff, stickiness

    def record(source, target, contact_type, duration, overlap, distance,
               repeat_count, depth, cleaning, raw, coeff, stickiness, efficacy,
               seconds_since_exposure):
        nonlocal event_index
        src_count = state.contact_count.get(source, 0)
        tgt_count = state.contact_count.get(target, 0)
        prev_time = state.last_time.get(target)
        time_since_last = 0.0 if prev_time is None else max(0.0, sim_time - prev_time)
        is_allergen = 1 if source in ALLERGEN_SOURCES else 0
        if source in ALLERGEN_SOURCES:
            source_risk = 1.0
        elif source == CLEANING_SUPPLY_LABEL:
            source_risk = 0.0
        else:
            source_risk = state.observed_risk.get(source, 0.0)

        row = {
            "scenario_id": scenario_id,
            "event_index": event_index,
            "scenario_type": scenario_type,
            "contact_type": contact_type,
            # ---- model features (FEATURE_ORDER) ----
            "source_object": source,
            "target_object": target,
            "source_current_risk": round(source_risk, 4),
            "target_previous_risk": round(state.observed_risk.get(target, 0.0), 4),
            "is_source_allergen": is_allergen,
            "contact_duration": round(duration, 3),
            "bbox_overlap_ratio": round(overlap, 4),
            "normalized_distance": round(distance, 4),
            "time_since_last_contact": round(time_since_last, 3),
            "source_contact_count": src_count,
            "target_contact_count": tgt_count,
            "propagation_depth": depth,
            "cleaning_detected": 1 if cleaning else 0,
            "repeated_contact_count": repeat_count,
            "seconds_since_source_exposure": round(seconds_since_exposure, 3),
            # ---- label ----
            "risk_class_id": _bucket(raw, rng),
            # ---- transparency-only latents (never features) ----
            "latent_regime": regime,
            "latent_target_contamination": round(raw, 4),
            "latent_transfer_coeff": round(coeff, 4),
            "latent_stickiness": round(stickiness, 4),
            "latent_cleaning_efficacy": round(efficacy, 4) if efficacy is not None else "",
        }
        row["risk_class"] = RISK_ID_TO_CLASS[row["risk_class_id"]]
        events.append(row)

        # Update running state for the target (noisy observation carried on).
        state.contact_count[source] = src_count + 1
        state.contact_count[target] = tgt_count + 1
        state.last_time[target] = sim_time
        if contact_type != "unrelated":
            state.latent[target] = raw
            state.observed_risk[target] = float(np.clip(raw + rng.normal(0.0, obs_noise), 0.0, 1.0))
            state.depth[target] = depth
            if target not in state.contaminated:
                state.contaminated.append(target)
        event_index += 1

    # ---- Event 0 ----------------------------------------------------------
    if preset["allergen_start"]:
        source = ALLERGEN_SOURCES[int(rng.integers(len(ALLERGEN_SOURCES)))]
        first_pool = preset.get("first_target", GENERATION_OBJECTS)
        target = first_pool[int(rng.integers(len(first_pool)))]
        duration, overlap, distance = _draw_geometry(rng, preset["duration"])
        raw, coeff, stick = transfer_label(0.0, True, duration, overlap, distance, 0, 0, 0.0)
        record(source, target, "direct_source", duration, overlap, distance,
               0, 0, False, raw, coeff, stick, None, 0.0)
    else:
        # Unrelated scene: two clean objects touch, ~zero risk.
        a, b = rng.choice(GENERATION_OBJECTS, size=2, replace=False)
        duration, overlap, distance = _draw_geometry(rng, preset["duration"])
        raw = float(np.clip(rng.normal(0.0, 0.03), 0.0, 1.0))
        record(str(a), str(b), "unrelated", duration, overlap, distance,
               0, 0, False, raw, 0.0, 0.0, None, 0.0)

    # ---- Further events ---------------------------------------------------
    num_extra = int(rng.integers(preset["extra"][0], preset["extra"][1] + 1))
    pair_counts = {}
    for _ in range(num_extra):
        sim_time += float(rng.exponential(TIME_GAP_MEAN_SECONDS))
        seconds_since_exposure = sim_time  # event 0 (source entry) is t=0
        action = _weighted_choice(rng, preset["actions"])
        if action in ("chain", "branch", "repeat") and not state.contaminated:
            action = "unrelated"

        duration, overlap, distance = _draw_geometry(rng, preset["duration"])

        if action == "clean" and state.contaminated:
            target = state.contaminated[int(rng.integers(len(state.contaminated)))]
            efficacy = float(rng.uniform(*CLEAN_EFFICACY_BANDS[preset["clean_efficacy"]]))
            residual = state.observed_risk.get(target, 0.0) * (1.0 - efficacy)
            raw = float(np.clip(residual + rng.normal(0.0, 0.02), 0.0, 1.0))
            depth = state.depth.get(target, 0)
            record(CLEANING_SUPPLY_LABEL, target, "clean", duration, overlap, distance,
                   0, depth, True, raw, 0.0, 0.0, efficacy, seconds_since_exposure)

        elif action == "unrelated":
            a, b = rng.choice(GENERATION_OBJECTS, size=2, replace=False)
            raw = float(np.clip(rng.normal(0.0, 0.03), 0.0, 1.0))
            record(str(a), str(b), "unrelated", duration, overlap, distance,
                   0, 0, False, raw, 0.0, 0.0, None, seconds_since_exposure)

        elif action == "repeat":
            source = state.contaminated[int(rng.integers(len(state.contaminated)))]
            candidates = [o for o in GENERATION_OBJECTS if o != source]
            target = candidates[int(rng.integers(len(candidates)))]
            pair = (source, target)
            repeat_count = pair_counts.get(pair, 0) + 1
            pair_counts[pair] = repeat_count
            depth = state.depth.get(source, 0) + 1
            raw, coeff, stick = transfer_label(
                state.observed_risk.get(source, 0.0), False, duration, overlap,
                distance, repeat_count, depth, seconds_since_exposure)
            record(source, target, "repeat", duration, overlap, distance,
                   repeat_count, depth, False, raw, coeff, stick, None, seconds_since_exposure)

        else:  # chain / branch
            source = state.contaminated[int(rng.integers(len(state.contaminated)))]
            if action == "branch":
                fresh = [o for o in GENERATION_OBJECTS
                         if o != source and o not in state.contaminated]
                pool = fresh if fresh else [o for o in GENERATION_OBJECTS if o != source]
            else:
                pool = [o for o in GENERATION_OBJECTS if o != source]
            target = pool[int(rng.integers(len(pool)))]
            pair = (source, target)
            pair_counts[pair] = pair_counts.get(pair, 0)
            depth = state.depth.get(source, 0) + 1
            raw, coeff, stick = transfer_label(
                state.observed_risk.get(source, 0.0), False, duration, overlap,
                distance, 0, depth, seconds_since_exposure)
            record(source, target, action, duration, overlap, distance,
                   0, depth, False, raw, coeff, stick, None, seconds_since_exposure)

    meta = {
        "scenario_id": scenario_id,
        "scenario_type": scenario_type,
        "latent_regime": regime,
        "num_events": len(events),
        "latent_depth_decay": round(depth_decay, 4),
        "latent_time_decay_rate": round(time_decay_rate, 4),
        "latent_obs_noise": round(obs_noise, 4),
        "latent_label_noise": round(label_noise, 4),
    }
    return events, meta


def generate_dataset(num_scenarios: int = DEFAULT_NUM_SCENARIOS, seed: int = DEFAULT_SEED):
    """Generate the full dataset. Returns (events_df, scenario_meta_df)."""
    rng = np.random.default_rng(seed)
    types = list(SCENARIO_PRESETS.keys())
    weights = np.array([SCENARIO_PRESETS[t]["weight"] for t in types], dtype=float)
    weights = weights / weights.sum()

    all_events, all_meta = [], []
    for scenario_id in range(num_scenarios):
        scenario_type = types[int(rng.choice(len(types), p=weights))]
        events, meta = generate_scenario(scenario_id, scenario_type, rng)
        all_events.extend(events)
        all_meta.append(meta)

    import pandas as pd

    events_df = pd.DataFrame(all_events)
    meta_df = pd.DataFrame(all_meta)

    # Stable, self-documenting column order: keys, features, label, latents.
    lead = ["scenario_id", "event_index", "scenario_type", "contact_type"]
    label_cols = ["risk_class_id", "risk_class"]
    latent_cols = [c for c in events_df.columns if c.startswith("latent_")]
    ordered = lead + list(FEATURE_ORDER) + label_cols + latent_cols
    events_df = events_df[ordered]
    return events_df, meta_df


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate synthetic risk-event training data.")
    parser.add_argument("--num-scenarios", type=int, default=DEFAULT_NUM_SCENARIOS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-rows", type=int, default=20000,
                        help="Fail if fewer than this many event rows are generated.")
    args = parser.parse_args(argv)

    events_df, meta_df = generate_dataset(args.num_scenarios, args.seed)
    if len(events_df) < args.min_rows:
        raise SystemExit(
            f"Generated only {len(events_df)} rows (< {args.min_rows}). "
            f"Increase --num-scenarios.")

    os.makedirs(args.output_dir, exist_ok=True)
    events_path = os.path.join(args.output_dir, "risk_events.csv")
    meta_path = os.path.join(args.output_dir, "scenario_metadata.csv")
    events_df.to_csv(events_path, index=False)
    meta_df.to_csv(meta_path, index=False)

    dist = events_df["risk_class"].value_counts().reindex(["LOW", "MEDIUM", "HIGH"]).fillna(0).astype(int)
    print(f"Wrote {len(events_df)} events across {args.num_scenarios} scenarios "
          f"({meta_df['scenario_id'].nunique()} scenario groups).")
    print(f"  events   -> {events_path}")
    print(f"  metadata -> {meta_path}")
    print("Class distribution:")
    for label in ["LOW", "MEDIUM", "HIGH"]:
        print(f"  {label:<6} {dist[label]:>7}  ({dist[label] / len(events_df):.1%})")
    print("Scenario-type counts:")
    print(meta_df["scenario_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
