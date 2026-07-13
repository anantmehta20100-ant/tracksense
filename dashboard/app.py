"""Streamlit live dashboard: setup screen (pick user_allergen) -> live webcam
feed with tracked-object boxes, a live risk map, and a prominent exposure
alert banner.

Streamlit reruns the whole script on every interaction, so the running
pipeline (a LiveRunner) lives in st.session_state rather than as a local
variable. To get a "live" feed despite that rerun model, one frame is
processed per script run and st.rerun() is called at the end to loop --
the standard workaround for continuous webcam display in Streamlit. This is
a known tradeoff: a Stop click is only honored between reruns (i.e. after
the current frame finishes), not instantly.
"""

import time

import cv2
import streamlit as st

from config.allergens import ALLERGEN_TYPES, EXPOSURE_ALERT_RISK_THRESHOLD
from pipeline.live_runner import LiveRunner

RERUN_DELAY_SECONDS = 0.03  # throttles the rerun loop; not a true frame-rate cap

RISK_HIGH_THRESHOLD = EXPOSURE_ALERT_RISK_THRESHOLD
RISK_MEDIUM_THRESHOLD = 0.3


def _severity_label(risk: float) -> str:
    if risk >= RISK_HIGH_THRESHOLD:
        return "🔴 High"
    if risk >= RISK_MEDIUM_THRESHOLD:
        return "🟠 Medium"
    return "🟢 Low"


def _init_session_state():
    if "running" not in st.session_state:
        st.session_state.running = False
    if "runner" not in st.session_state:
        st.session_state.runner = None


def render_setup_screen():
    st.title("TrackSense")
    st.caption(
        "Research MVP: predicts food-allergen cross-contact risk propagation from a live "
        "camera feed. Not a certified food-safety device -- see limitations below."
    )

    with st.expander("Scientific honesty / limitations", expanded=False):
        st.markdown(
            "- Predicts **relative risk** from observed/synthetic interaction patterns, "
            "not measured allergen concentration.\n"
            "- Contact detection uses a **proximity/bounding-box heuristic**, not a trained "
            "gesture/grip model -- a disclosed, deliberate scope decision.\n"
            "- The risk model was trained on **synthetic** contact sequences, since no real "
            "labeled cross-contact dataset exists. Live *detection* is real; the risk "
            "*prediction* model was trained offline.\n"
            "- This is a research proof of concept, not a medical or safety device."
        )

    allergen = st.selectbox("Which allergen are you tracking for this session?", ALLERGEN_TYPES)

    if st.button("Start session", type="primary"):
        try:
            st.session_state.runner = LiveRunner(user_allergen=allergen)
            st.session_state.running = True
            st.rerun()
        except FileNotFoundError as error:
            st.error(str(error))


def render_main_screen():
    runner = st.session_state.runner

    st.title("TrackSense -- live session")
    st.caption(f"Watching for allergen: **{runner.user_allergen}**")

    if st.button("Stop session"):
        runner.stop_camera()
        st.session_state.running = False
        st.session_state.runner = None
        st.rerun()

    alert_placeholder = st.empty()
    video_col, risk_col = st.columns([2, 1])
    frame_placeholder = video_col.empty()
    risk_placeholder = risk_col.empty()

    result = runner.process_frame()
    if result is None:
        st.warning("No camera frame available.")
        return

    frame = result["frame"]
    _draw_overlays(frame, result["tracks"])
    frame_placeholder.image(frame, channels="BGR")

    if result["active_alerts"]:
        alert_placeholder.error(
            "🚨 EXPOSURE ALERT -- a food object with elevated risk for your allergen "
            "was near your mouth."
        )
    else:
        alert_placeholder.empty()

    with risk_placeholder.container():
        st.subheader("Live risk map")
        rows = [
            {
                "Track ID": track_id,
                "Allergen": allergen_type,
                "Risk": round(risk, 2),
                "Severity": _severity_label(risk),
            }
            for (track_id, allergen_type), risk in result["risk_state"].risk.items()
        ]
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.write("No tracked objects with risk yet.")

    time.sleep(RERUN_DELAY_SECONDS)
    st.rerun()


def _draw_overlays(frame, tracks):
    for track in tracks:
        x1, y1, x2, y2 = (int(v) for v in track.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{track.class_name}#{track.track_id}",
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )


def main():
    st.set_page_config(page_title="TrackSense", layout="wide")
    _init_session_state()

    if st.session_state.running and st.session_state.runner is not None:
        render_main_screen()
    else:
        render_setup_screen()


if __name__ == "__main__":
    main()
