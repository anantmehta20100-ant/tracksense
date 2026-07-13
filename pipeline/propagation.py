"""Propagation-chain explanations (Phase 11).

Turns the engine's per-object provenance (risk_chain of track_ids + root
allergen) into human-facing explanations of WHY a downstream object is at risk,
e.g. plate is risky because:

    nut_butter_jar -> cutlery -> bread -> plate

Wording is deliberately about PREDICTED DOWNSTREAM CROSS-CONTACT RISK, never
"peanut detected on plate" -- the system predicts relative risk, it does not
measure contamination (see README limitations).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.runtime_config import ALERT  # noqa: E402
from ml.risk_features import RISK_CLASS_TO_ID  # noqa: E402

ARROW = " → "   # " -> " for the UI


def resolve_chain(chain_ids, risk_map) -> list:
    """Map a chain of track_ids to [{track_id, class}] using current object
    state (class names). Unknown ids resolve to class '?' rather than dropping,
    so provenance is never silently lost."""
    resolved = []
    for track_id in chain_ids:
        state = risk_map.get(track_id)
        class_name = state["class_name"] if state else "?"
        resolved.append({"track_id": track_id, "class": class_name})
    return resolved


def chain_text(resolved) -> str:
    return ARROW.join(item["class"] for item in resolved)


def _is_elevated(state, min_class_id) -> bool:
    return RISK_CLASS_TO_ID.get(state.get("risk_class", "LOW"), 0) >= min_class_id


def build_explanation(state, risk_map, *, min_class: str = "MEDIUM"):
    """Explanation dict for one object, or None if it is not an elevated
    DOWNSTREAM risk (needs a real chain of length >= 2 rooted in an allergen)."""
    min_class_id = RISK_CLASS_TO_ID[min_class]
    if state.get("root_allergen_track_id") is None:
        return None
    chain_ids = state.get("risk_chain") or []
    if len(chain_ids) < 2:            # length-1 chain == the allergen source itself
        return None
    if not _is_elevated(state, min_class_id):
        return None

    resolved = resolve_chain(chain_ids, risk_map)
    return {
        "object": state["class_name"],
        "track_id": state["track_id"],
        "risk_class": state["risk_class"],
        "risk_score": state["risk_score"],
        "propagation_depth": state.get("propagation_depth", 0),
        "root_allergen_track_id": state["root_allergen_track_id"],
        "chain": resolved,
        "chain_text": chain_text(resolved),
        "note": "Predicted downstream cross-contact risk (relative, not a measured contamination level).",
    }


def build_explanations(risk_map, *, min_class: str = "MEDIUM") -> list:
    """All elevated downstream-risk explanations, highest risk first."""
    out = []
    for state in risk_map.values():
        exp = build_explanation(state, risk_map, min_class=min_class)
        if exp is not None:
            out.append(exp)
    out.sort(key=lambda e: e["risk_score"], reverse=True)
    return out


def build_alerts(risk_map) -> list:
    """Downstream objects that cross the alert threshold (HIGH class or score >=
    configured level). Built from explanations so an alert always carries its
    'why' chain."""
    alerts = []
    for exp in build_explanations(risk_map, min_class="MEDIUM"):
        state = risk_map[exp["track_id"]]
        is_high = ALERT.alert_on_high_class and state["risk_class"] == "HIGH"
        over_score = state["risk_score"] >= ALERT.alert_risk_score
        if is_high or over_score:
            alerts.append({
                "object": exp["object"], "track_id": exp["track_id"],
                "risk_class": exp["risk_class"], "risk_score": exp["risk_score"],
                "chain_text": exp["chain_text"], "chain": exp["chain"],
            })
    return alerts
