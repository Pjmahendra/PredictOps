"""SIMULATED live sensor stream (demo only — synthetic data, never mixed with real results).

The simulator emits 10-s readings continuously and cycles through phases:

    healthy (random 30–120 sim-min) → degrading (20–45 min, failure blend ramps 0→1)
    → failure (30–60 min) → maintenance (15–30 min, logger silent = new segment) → healthy (45–150 min) → ...

so a failure starts at a random, unannounced time.

How the readings are made (so they look like the compressor, not random noise):
  * "healthy" readings are taken from a randomly chosen, contiguous stretch of REAL normal
    MetroPT-3 operation (> 3 days from any reported failure, no logger gaps, not frozen);
  * "failure" readings are taken from a randomly chosen REAL reported failure window;
  * during degrading the two are blended: analog = (1-w)·normal + w·failure,
    digital = failure value with probability w; small Gaussian noise is added.
  * timestamps are re-stamped onto the simulation clock.

The output is therefore SYNTHETIC: it is a recombination of real recorded signals, not a
measurement of any machine. It is used only to demonstrate the pipeline reacting live.
The phase schedule (ground truth of the simulation) is exposed so detection delay can be seen.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

ANALOG = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature", "Motor_current"]
DIGITAL = ["COMP", "DV_eletric", "Towers", "MPG", "LPS", "Pressure_switch", "Oil_level", "Caudal_impulses"]
SENSORS = ANALOG + DIGITAL
STEP_S = 10                       # one reading every 10 s (the CSV's real sampling rate)
PER_MIN = 60 // STEP_S
WARMUP_MIN = 180                  # healthy history so 2 h rolling features are filled
BUFFER_MIN = 6 * 60               # raw readings kept for feature computation
HISTORY_MIN = 12 * 60             # per-minute results kept for the dashboard


@dataclass
class Phase:
    name: str          # healthy | degrading | failure | maintenance
    start: pd.Timestamp
    minutes: int

    @property
    def end(self) -> pd.Timestamp:
        return self.start + pd.Timedelta(minutes=self.minutes)


def _contiguous_blocks(raw: pd.DataFrame, min_rows: int) -> list[tuple[int, int]]:
    """(start_idx, end_idx) of runs with no gap > 30 s and analog sensors not frozen."""
    gap = raw.timestamp.diff().dt.total_seconds().fillna(STEP_S) > 30
    frozen = raw[ANALOG].diff().abs().sum(axis=1).rolling(60, min_periods=1).sum() == 0
    brk = (gap | frozen).to_numpy()
    idx = np.flatnonzero(brk)
    bounds = np.concatenate([[0], idx, [len(raw)]])
    return [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if b - a >= min_rows]


class SignalSource:
    """Random contiguous pieces of real recorded data, consumed row by row."""

    def __init__(self, frame: pd.DataFrame, blocks: list[tuple[int, int]], rng: np.random.Generator):
        self.values = frame[SENSORS].to_numpy(np.float64)
        self.blocks, self.rng = blocks, rng
        self._jump()

    def _jump(self):
        a, b = self.blocks[self.rng.integers(len(self.blocks))]
        self.pos, self.stop = int(self.rng.integers(a, max(a + 1, b - PER_MIN * 30))), b

    def next(self) -> np.ndarray:
        if self.pos >= self.stop:
            self._jump()
        v = self.values[self.pos]; self.pos += 1
        return v


class StreamSimulator:
    def __init__(self, raw: pd.DataFrame, failure_events: list[dict], predictor, speed: float = 1.0,
                 seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.predictor, self.speed = predictor, float(speed)
        raw = raw[["timestamp"] + SENSORS].sort_values("timestamp").reset_index(drop=True)
        fail_mask = np.zeros(len(raw), bool); far_mask = np.ones(len(raw), bool)
        for e in failure_events:
            s, t = pd.Timestamp(e["start"]), pd.Timestamp(e["end"])
            fail_mask |= raw.timestamp.between(s, t).to_numpy()
            far_mask &= ~raw.timestamp.between(s - pd.Timedelta(days=3), t + pd.Timedelta(days=3)).to_numpy()
        normal, failure = raw[far_mask].reset_index(drop=True), raw[fail_mask].reset_index(drop=True)
        self.normal = SignalSource(normal, _contiguous_blocks(normal, PER_MIN * 120), self.rng)
        self.failure = SignalSource(failure, _contiguous_blocks(failure, PER_MIN * 30), self.rng)
        self.noise = normal[ANALOG].std().to_numpy() * 0.01

        self.clock = pd.Timestamp.now().floor("min") - pd.Timedelta(minutes=WARMUP_MIN)
        self.buffer: deque = deque(maxlen=BUFFER_MIN * PER_MIN)
        self.results: deque = deque(maxlen=HISTORY_MIN)
        self.phases: list[Phase] = []
        self._new_phase("healthy", WARMUP_MIN + int(self.rng.integers(30, 121)))
        self.lock = threading.Lock()
        self._stop = threading.Event()
        self.thread: threading.Thread | None = None
        self._emit(WARMUP_MIN)

    # ------------------------------------------------------------- phases
    def _new_phase(self, name: str, minutes: int):
        self.phases.append(Phase(name, self.clock, minutes))
        self.phases = self.phases[-50:]

    def _advance_phase(self):
        nxt = {"healthy": ("degrading", (20, 46)), "degrading": ("failure", (30, 61)),
               "failure": ("maintenance", (15, 31)), "maintenance": ("healthy", (45, 151))}
        name, (lo, hi) = nxt[self.phases[-1].name]
        self._new_phase(name, int(self.rng.integers(lo, hi)))

    def _blend_weight(self) -> float:
        p = self.phases[-1]
        f = (self.clock - p.start) / pd.Timedelta(minutes=p.minutes)
        return {"healthy": 0.0, "degrading": f, "failure": 1.0, "maintenance": 0.0}[p.name]

    # ------------------------------------------------------------- generation
    def _emit(self, minutes: int):
        """Generate `minutes` of 10-s readings, then re-run the inference pipeline."""
        for _ in range(minutes * PER_MIN):
            if self.clock >= self.phases[-1].end:
                self._advance_phase()
            if self.phases[-1].name == "maintenance":      # compressor stopped for repair: no readings
                self.clock += pd.Timedelta(seconds=STEP_S)  # (the gap starts a new data segment, as
                continue                                    #  after real maintenance stops)
            w = float(np.clip(self._blend_weight(), 0, 1))
            n, f = self.normal.next(), self.failure.next()
            k = len(ANALOG)
            analog = (1 - w) * n[:k] + w * f[:k] + self.rng.normal(0, self.noise)
            digital = np.where(self.rng.random(len(DIGITAL)) < w, f[k:], n[k:])
            self.clock += pd.Timedelta(seconds=STEP_S)
            self.buffer.append((self.clock, *analog, *digital, self.phases[-1].name))
        self._score(minutes)

    def _score(self, new_minutes: int):
        df = pd.DataFrame(list(self.buffer), columns=["timestamp"] + SENSORS + ["phase"])
        raw = df[["timestamp"] + SENSORS].assign(timestamp=df.timestamp.astype(str))
        res = self.predictor.predict_raw(raw)
        phase = df.set_index("timestamp").phase
        res["sim_phase"] = phase.reindex(res.timestamp, method="ffill").to_numpy()
        for r in res.tail(new_minutes).to_dict("records"):
            self.results.append(r)

    # ------------------------------------------------------------- thread control
    def step(self, minutes: int = 1):
        with self.lock:
            self._emit(minutes)

    def _loop(self):
        acc = 0.0
        while not self._stop.is_set():
            t0 = time.monotonic()
            acc += self.speed
            if acc >= 1:
                n = int(acc); acc -= n
                try:
                    self.step(n)
                except Exception:          # keep streaming; the next tick retries
                    import logging; logging.getLogger("predictops.sim").exception("simulation step failed")
            self._stop.wait(max(0.0, 1.0 - (time.monotonic() - t0)))

    def start(self):
        if not (self.thread and self.thread.is_alive()):
            self._stop.clear()
            self.thread = threading.Thread(target=self._loop, name="predictops-sim", daemon=True)
            self.thread.start()

    def stop(self):
        self._stop.set()
        if self.thread:
            self.thread.join(timeout=5)

    @property
    def running(self) -> bool:
        return bool(self.thread and self.thread.is_alive())

    # ------------------------------------------------------------- read-out
    def snapshot(self, minutes: int = 180) -> dict:
        with self.lock:
            rows = list(self.results)[-minutes:]
            phases = [{"phase": p.name, "start": p.start.isoformat(), "end": min(p.end, self.clock).isoformat()}
                      for p in self.phases if p.start <= self.clock]
            cur = self.phases[-1].name
        return {"mode": "SIMULATED (synthetic data)", "running": self.running, "speed": self.speed,
                "sim_time": rows[-1]["timestamp"].isoformat() if rows else None, "current_phase": cur,
                "phases": phases, "rows": rows}
