# 14 — Live Simulation (synthetic data stream)

**What it is:** a demo mode that streams **synthetic** sensor readings continuously. At random, unannounced
times the stream drifts from healthy operation into an air-leak failure, then recovers after "maintenance".
The full PredictOps pipeline (features → One-Class SVM + XGBoost → health engine) scores it live.

> ⚠ This is **simulated data**, not a real machine and not an evaluation result. It is labelled
> "SIMULATED" in the API (`mode` field) and on the dashboard page. Simulated runs are kept in memory
> only — they are never written to the database or mixed with the real MetroPT-3 results.

## Phase cycle (durations random each time)
| Phase | Duration (sim-minutes) | What the readings are |
|---|---|---|
| healthy | 45–150 (first one includes a 3 h warm-up) | real normal MetroPT-3 operation (> 3 days from any reported failure, no gaps, not frozen) |
| degrading | 20–45 | blend: analog = (1−w)·normal + w·failure, w ramps 0 → 1; digital = failure value with prob. w |
| failure | 30–60 | real readings from a reported failure window (events 1–4) |
| maintenance | 15–30 | **no readings** (compressor stopped; the gap starts a new data segment, like real repairs) |

Small Gaussian noise (1 % of each sensor's std) is added; timestamps are put on a simulated clock.
Real recorded signals are recombined so the stream looks like the compressor rather than random noise.

## How to use
* **Docker:** starts automatically (`SIMULATION_AUTOSTART=true`, `SIMULATION_SPEED=2` in `docker-compose.yml`).
* **Manual:** start the API, open the dashboard → *Monitor → Live Simulation* → **▶ Start**. Or set
  `SIMULATION_AUTOSTART=true` before starting `uvicorn`.
* Speed = simulated minutes per real second (2 → one full failure cycle every ~1–2 real minutes).
* Toggle *Show injected phases* to hide the ground truth and see whether you can spot the failure from
  the model output alone.

## API
| Endpoint | Purpose |
|---|---|
| `POST /simulation/start?speed=2&seed=&reset=` | start / resume (a `seed` or `reset=true` creates a fresh stream) |
| `POST /simulation/stop` | pause |
| `GET /simulation/state?minutes=180` | latest per-minute assessments + sensors + phase schedule |

## Verified behaviour (`tests/test_simulation.py`, seed 1, 400 sim-minutes)
| Injected phase | Mean health | Critical minutes | Anomaly alert |
|---|---|---|---|
| healthy | 97.4 | 0 % | 1 % |
| degrading | 78.1 | 14 % | 24 % |
| failure | 23.5 | 92 % | 99 % |

The pipeline flags each injected failure, usually during the late degrading phase. Because the failure
segments are **real** failure recordings that the models were trained / validated on (events 1–3), this
demonstrates the app working, **not** model accuracy — use docs/05–07 for that.

Code: `src/predictops/simulation/simulator.py`, `backend/routes/api.py` (simulation section),
`frontend/views/simulation.py`.
