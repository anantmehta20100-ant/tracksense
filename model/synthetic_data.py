"""Generates randomized, labeled synthetic contact-event sequences for
training the GRU risk-propagation model (see AGENTS.md "Dataset strategy").

Each sequence is a mini kitchen scene: an allergen source touches a first
object, then a random-length chain of further contacts follows -- some are
direct continuations of the contamination chain, some are unrelated
(negative) contacts with no connection to it, and some are cleaning events
that reset an object's risk. Within one sequence, an object class name (e.g.
"cutlery") stands in for one concrete instance of that object in the scene --
sequences don't model multiple simultaneous objects of the same class.

Risk label rule (documented, not a black box):
    - The allergen source always starts at risk 1.0.
    - Each hop down a contamination chain multiplies the parent's risk by
      RULE_BASED_DECAY_RATE (config/allergens.py), plus small Gaussian noise,
      clipped to [0, 1]. This is the same rule RuleBasedDecayBaseline uses
      without noise -- the synthetic data is intentionally decay-shaped so
      there is real signal to learn, not pure noise.
    - A cleaning event resets the target object's risk to
      CLEANING_RISK_RESET_VALUE plus a small noise term, and its propagation
      depth resets to 0.
    - An unrelated contact involves two objects with no path back to the
      allergen source; its target risk label is ~0 plus small noise.
"""

import argparse
import os
import random

import pandas as pd

from config.allergens import (
    ALLERGEN_SOURCE_CLASSES,
    CLEANING_RISK_RESET_VALUE,
    OBJECT_CLASSES,
    RULE_BASED_DECAY_RATE,
    get_allergen_type,
)

# Sentinel for events with no connection to any allergen contamination chain
# (e.g. two clean, unrelated objects touching). Kept as an explicit string
# rather than NaN so it's a normal category the GRU's embedding can learn,
# consistent with allergen_type being a first-class field everywhere
# (AGENTS.md "the one architecture decision that makes the stretch goal cheap").
NO_ALLERGEN_LABEL = "none"

SOURCE_OBJECTS = list(ALLERGEN_SOURCE_CLASSES.keys())
NON_SOURCE_OBJECTS = [c for c in OBJECT_CLASSES if c not in ALLERGEN_SOURCE_CLASSES]

CHAIN_NOISE_STD = 0.05
CLEANING_NOISE_STD = 0.02
UNRELATED_NOISE_STD = 0.03

CLEANING_EVENT_PROBABILITY = 0.15
UNRELATED_EVENT_PROBABILITY = 0.25

MIN_CHAIN_EVENTS = 2
MAX_CHAIN_EVENTS = 8
TIME_GAP_MEAN_SECONDS = 5.0

DEFAULT_NUM_SEQUENCES = 2000

CLEANING_SUPPLY_LABEL = "cleaning_supply"


def _clip(value: float) -> float:
    return min(1.0, max(0.0, value))


def _noisy(value: float, std: float) -> float:
    return _clip(value + random.gauss(0.0, std))


def generate_sequence(sequence_id: int):
    """Generates one labeled contact sequence. Returns a list of event dicts."""
    events = []

    risk_state = {}  # object class name -> current risk score
    depth_state = {}  # object class name -> current propagation depth
    allergen_state = {}  # object class name -> allergen_type carried by it
    contaminated_objects = []  # objects with risk_state > 0, in touch order

    event_index = 0

    # Event 0: allergen source directly touches a first object.
    source_object = random.choice(SOURCE_OBJECTS)
    target_object = random.choice(NON_SOURCE_OBJECTS)
    source_risk = 1.0
    allergen_type = get_allergen_type(source_object)
    target_risk_label = _noisy(source_risk * RULE_BASED_DECAY_RATE, CHAIN_NOISE_STD)

    events.append(
        {
            "sequence_id": sequence_id,
            "event_index": event_index,
            "source_object": source_object,
            "target_object": target_object,
            "source_risk": source_risk,
            "contact_type": "direct_source",
            "time_since_previous_contact": 0.0,
            "cleaning_event": False,
            "propagation_depth": 1,
            "allergen_type": allergen_type,
            "target_risk_label": target_risk_label,
        }
    )
    risk_state[target_object] = target_risk_label
    depth_state[target_object] = 1
    allergen_state[target_object] = allergen_type
    contaminated_objects.append(target_object)

    num_further_events = random.randint(MIN_CHAIN_EVENTS, MAX_CHAIN_EVENTS)

    for _ in range(num_further_events):
        event_index += 1
        time_since_previous_contact = round(random.expovariate(1.0 / TIME_GAP_MEAN_SECONDS), 2)
        roll = random.random()

        if roll < CLEANING_EVENT_PROBABILITY and contaminated_objects:
            target_object = random.choice(contaminated_objects)
            source_object = CLEANING_SUPPLY_LABEL
            source_risk = risk_state[target_object]
            allergen_type = allergen_state[target_object]
            cleaning_event = True
            propagation_depth = 0
            target_risk_label = _noisy(CLEANING_RISK_RESET_VALUE, CLEANING_NOISE_STD)
            contact_type = "cleaning"

            risk_state[target_object] = target_risk_label
            depth_state[target_object] = propagation_depth

        elif roll < CLEANING_EVENT_PROBABILITY + UNRELATED_EVENT_PROBABILITY:
            clean_candidates = [o for o in NON_SOURCE_OBJECTS if o not in risk_state]
            if len(clean_candidates) < 2:
                clean_candidates = NON_SOURCE_OBJECTS
            source_object, target_object = random.sample(clean_candidates, 2)
            source_risk = 0.0
            allergen_type = NO_ALLERGEN_LABEL
            cleaning_event = False
            propagation_depth = 0
            target_risk_label = _noisy(0.0, UNRELATED_NOISE_STD)
            contact_type = "unrelated"
            # Deliberately not added to risk_state/contaminated_objects: this
            # object has no connection to the contamination chain.

        else:
            source_object = random.choice(contaminated_objects)
            source_risk = risk_state[source_object]
            allergen_type = allergen_state[source_object]
            candidates = [o for o in OBJECT_CLASSES if o != source_object]
            target_object = random.choice(candidates)
            cleaning_event = False
            propagation_depth = depth_state[source_object] + 1
            target_risk_label = _noisy(source_risk * RULE_BASED_DECAY_RATE, CHAIN_NOISE_STD)
            contact_type = "chain"

            risk_state[target_object] = target_risk_label
            depth_state[target_object] = propagation_depth
            if target_object not in contaminated_objects:
                contaminated_objects.append(target_object)

        allergen_state[target_object] = allergen_type

        events.append(
            {
                "sequence_id": sequence_id,
                "event_index": event_index,
                "source_object": source_object,
                "target_object": target_object,
                "source_risk": source_risk,
                "contact_type": contact_type,
                "time_since_previous_contact": time_since_previous_contact,
                "cleaning_event": cleaning_event,
                "propagation_depth": propagation_depth,
                "allergen_type": allergen_type,
                "target_risk_label": target_risk_label,
            }
        )

    return events


def generate_dataset(num_sequences: int = DEFAULT_NUM_SEQUENCES, seed: int = None) -> pd.DataFrame:
    if seed is not None:
        random.seed(seed)

    all_events = []
    for sequence_id in range(num_sequences):
        all_events.extend(generate_sequence(sequence_id))

    return pd.DataFrame(all_events)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic contact-event training data.")
    parser.add_argument("--num-sequences", type=int, default=DEFAULT_NUM_SEQUENCES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join("data", "synthetic", "sequences.csv"),
    )
    args = parser.parse_args()

    dataset = generate_dataset(num_sequences=args.num_sequences, seed=args.seed)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    dataset.to_csv(args.output, index=False)
    print(f"Wrote {len(dataset)} events across {args.num_sequences} sequences to {args.output}")
