"""Live overlap reporter for the running TrackSense backend.

Polls GET /api/snapshot and prints a report line whenever two detected bounding
boxes overlap: both the moment a pair STARTS overlapping (from active_contacts)
and each COMPLETED contact event (from the timeline, which carries overlap_ratio
and the RF risk verdict). Read-only -- it does not touch the pipeline or camera.
"""

from __future__ import annotations

import json
import time
import urllib.request

URL = "http://127.0.0.1:5000/api/snapshot"
POLL_S = 0.25


def _snapshot():
    with urllib.request.urlopen(URL, timeout=2) as r:
        return json.load(r)


def main():
    seen_events = set()      # completed contact event_ids already reported
    active_pairs = set()     # pairs currently overlapping (already announced)
    print("Overlap monitor started -- watching for bounding-box overlaps...\n")

    while True:
        try:
            snap = _snapshot()
        except Exception as exc:  # backend down / restarting
            print(f"[monitor] snapshot unavailable: {exc}")
            time.sleep(POLL_S)
            continue

        f = snap.get("frame_index")

        # 1) Live overlaps happening right now.
        now_pairs = set()
        for c in snap.get("active_contacts", []):
            a = c.get("source_track_id", c.get("a_track_id"))
            b = c.get("target_track_id", c.get("b_track_id"))
            pair = tuple(sorted((str(a), str(b))))
            now_pairs.add(pair)
            if pair not in active_pairs:
                ov = c.get("overlap_ratio")
                sc = c.get("source_class", "?")
                tc = c.get("target_class", "?")
                extra = f" overlap={ov:.2f}" if isinstance(ov, (int, float)) else ""
                print(f"[frame {f}] OVERLAP START: {sc}#{a} <-> {tc}#{b}{extra}")
        active_pairs = now_pairs

        # 2) Completed contact events (final overlap ratio + risk verdict).
        for e in snap.get("timeline", []):
            if e.get("type") != "contact":
                continue
            eid = e.get("event_id")
            if eid in seen_events:
                continue
            seen_events.add(eid)
            print(
                f"[frame {e.get('frame_index')}] CONTACT #{eid}: "
                f"{e.get('source_class')}#{e.get('source_track_id')} <-> "
                f"{e.get('target_class')}#{e.get('target_track_id')} | "
                f"overlap={e.get('overlap_ratio')} dur={e.get('duration')}s | "
                f"risk={e.get('risk_class')} ({e.get('risk_score')})"
            )

        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
