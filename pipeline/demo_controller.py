"""Demo-mode controller (Phase 14).

Plays a scripted mock scenario through the real RiskPipeline over (simulated)
time, so a viewer watches risk PROPAGATE frame by frame rather than appearing all
at once. It is deliberately steppable:

  * step()              -- advance exactly one frame, return the fresh snapshot
  * run_to_completion() -- advance to the end (used by tests; fully synchronous)

The backend runs step() from a background thread on a wall-clock timer for the
live UI; tests call run_to_completion() for a deterministic, thread-free result.
Nothing here fabricates risk -- every number comes from the RF via RiskPipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.runtime_config import DEMO  # noqa: E402
from pipeline.risk_pipeline import RiskPipeline  # noqa: E402
from vision.mock_detection_source import MockDetectionSource, list_scenarios  # noqa: E402


class DemoController:
    def __init__(self, scenario: str = None, *, model_path=None, fps: float = None,
                 speed: float = None, seed: int = None):
        self.scenario = scenario or DEMO.scenario
        self.fps = float(fps if fps is not None else DEMO.fps)
        self.speed = float(speed if speed is not None else DEMO.speed)
        self.seed = int(seed if seed is not None else DEMO.seed)
        self._model_path = model_path
        self._build()

    def _build(self):
        self.pipeline = RiskPipeline(model_path=self._model_path, source_kind="mock")
        self.source = MockDetectionSource(self.scenario, fps=self.fps, seed=self.seed)
        self._frames = list(self.source.frames())
        self.cursor = 0
        self.status = "idle"      # idle | running | finished

    # -- controls ----------------------------------------------------------
    def reset(self, scenario: str = None):
        if scenario is not None:
            if scenario not in list_scenarios():
                raise ValueError(f"unknown scenario {scenario!r}")
            self.scenario = scenario
        self._build()
        return self.snapshot()

    def set_speed(self, speed: float):
        self.speed = max(0.1, float(speed))

    @property
    def total_frames(self) -> int:
        return len(self._frames)

    @property
    def finished(self) -> bool:
        return self.status == "finished"

    def next_delay_seconds(self) -> float:
        """Wall-clock delay the live driver should wait before the next step
        (frame period / speed). 0 when finished."""
        if self.status == "finished" or self.cursor >= len(self._frames):
            return 0.0
        return (1.0 / self.fps) / max(0.1, self.speed)

    def step(self) -> dict:
        """Advance one frame (or finalize). Returns the snapshot after stepping."""
        if self.cursor >= len(self._frames):
            if self.status != "finished":
                self.pipeline.finish()
                self.status = "finished"
            return self.snapshot()
        self.status = "running"
        frame = self._frames[self.cursor]
        self.pipeline.process_frame(frame)
        self.cursor += 1
        if self.cursor >= len(self._frames):
            self.pipeline.finish()
            self.status = "finished"
        return self.snapshot()

    def run_to_completion(self) -> dict:
        while self.status != "finished":
            self.step()
        return self.snapshot()

    def clean(self, track_id, timestamp=None):
        return self.pipeline.clean_track(track_id, timestamp=timestamp)

    # -- reporting ---------------------------------------------------------
    def snapshot(self) -> dict:
        snap = self.pipeline.snapshot()
        snap.update({
            "scenario": self.scenario,
            "status": self.status,
            "cursor": self.cursor,
            "total_frames": self.total_frames,
            "progress": round(self.cursor / self.total_frames, 3) if self.total_frames else 1.0,
            "fps": self.fps,
            "speed": self.speed,
            "available_scenarios": list_scenarios(),
        })
        return snap


if __name__ == "__main__":
    for scen in list_scenarios():
        ctl = DemoController(scen)
        snap = ctl.run_to_completion()
        print(f"\n=== {scen} (frames={ctl.total_frames}) ===")
        for obj in snap["objects"]:
            print(f"  {obj['class_name']:<16}{obj['risk_class']:<7}"
                  f"score={obj['risk_score']:.3f} chain={obj['risk_chain']}")
        for exp in snap["explanations"]:
            print(f"  WHY {exp['object']}: {exp['chain_text'].encode('ascii','replace').decode()}")
