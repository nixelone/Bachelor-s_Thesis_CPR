# Visual Inspection of Chest Compressions During CPR

**Bachelor's Thesis — Charles University, Faculty of Mathematics and Physics, 2026**
**Author:** Nikolas Ján Hamarik | **Supervisor:** Ing. Adam Novozámský, Ph.D.

A markerless, server-side computer vision pipeline that measures chest compression frequency directly from smartphone video — no contact hardware required. Designed for integration into [Záchranka](https://www.zachranka.app/), the official Czech emergency application.

---

## Overview

Real-time CPR quality monitoring normally requires specialized accelerometer hardware. This pipeline replaces that hardware with a standard smartphone camera feed, processing video on a backend server and streaming per-second BPM telemetry to an emergency dispatcher's interface.

**Core algorithmic stages:**

1. **YOLO11x-pose** — extracts upper-body keypoints (shoulders, elbows, wrists) from every frame
2. **Lucas-Kanade optical flow** — tracks static background features to mathematically subtract camera drift from keypoint coordinates
3. **Temporal buffering** — constructs 1D vertical displacement signals in a 2.5-second rolling window
4. **Activity pre-filtering** — ranks subjects by vertical variance; only the top 3 are passed to CWT
5. **Continuous Wavelet Transform (CWT)** — extracts instantaneous frequency and a periodicity score from the non-stationary signal
6. **Geometric heuristic gating** — disqualifies bystanders using elbow angle, wrist spread, and face-to-wrist distance
7. **Temporal state machine** — applies stability windows and grace periods; outputs a 1 Hz nullable integer BPM stream

---

## Key Results (33-video custom dataset)

| Experiment | Condition | Uptime | MAE |
|---|---|---|---|
| Spatial robustness | Realistic handheld drift | ≥ 98 % | < 2.4 BPM |
| Spatial robustness | Extreme panic camera shake | 45–97 % | < 4.0 BPM |
| Non-stationary tracking | Fatigue curve (120 → 80 BPM) | 100 % | ~1.5 BPM |
| Non-stationary tracking | Adrenaline spike (90 → 135 BPM) | 98 % | < 3.3 BPM |
| Bystander isolation | All multi-subject scenarios | — | **0 % false-positive rate** |

Processing throughput on an RTX 5070 Ti: ~10 FPS / RTF ≈ 2.4–2.9 (real-time achievable via frame-dropping or enterprise GPU).

---

## Repository Structure

```
.
├── Main.py                    # Full pipeline (entry point)
├── requirements.txt
│
├── experiments/               # Per-experiment runner scripts
├── experiment_results/        # Raw JSON telemetry outputs from all 33 videos
├── ground_truth/              # Ground-truth BPM annotations (metronome tracks)
├── prototypes/                # Earlier pipeline iterations (weighted heuristic, probabilistic model)
├── helper_scripts/            # Evaluation utilities: MAE calculator, confusion matrix, plotter
├── figures/                   # Thesis figures and result charts
│
└── yolo_models/               # Place yolo11x-pose.pt here (see Setup)
```

> **Note on dataset videos:** The evaluation videos are not included in this repository due to size and privacy constraints. The JSON telemetry outputs and ground-truth annotations in `experiment_results/` and `ground_truth/` are sufficient to fully reproduce all reported metrics.

---

## Setup

**Requirements:** Python 3.10+, CUDA-capable GPU recommended.

```bash
git clone https://github.com/nixelone/Bachelor-s_Thesis_CPR
cd Bachelor-s_Thesis_CPR
pip install -r requirements.txt
```

**Download the YOLO model:**

```bash
# The model is downloaded automatically by Ultralytics on first run, or manually:
python -c "from ultralytics import YOLO; YOLO('yolo11x-pose.pt')"
# Move the downloaded file into yolo_models/
```

---

## Usage

```bash
python Main.py
```

By default, `main()` is called with `buffer_seconds=2.5`. Edit the `main()` call at the bottom of `Main.py` to point to your video source and output paths:

```python
# Inside Main.py — configure these before running:
source          = "your_video.mp4"
output_json_path = "output.json"
save_output     = True          # set False to skip annotated video rendering
buffer_seconds  = 2.5           # try 2.0 (faster, noisier) or 3.0 (smoother, slower)
```

### Output

The pipeline produces two outputs:

**1. Annotated video** (`cpr_analysis_output.mp4`) — overlays bounding boxes, keypoint skeleton, heuristic values, and a colour-coded live BPM readout.

**2. JSON telemetry log:**

```json
{
  "description": "CPR compression rate per second",
  "videoFile": "emergency_feed_01.mp4",
  "unit": "compressions/min",
  "data": [
    { "second": 0, "rate": null,  "performer_id": null },
    { "second": 1, "rate": 108,   "performer_id": 1    }
  ]
}
```

`rate` is `null` during occlusions or before the 2.5-second buffer fills. `performer_id` is a stable integer that persists across rescuer swaps.

---

## Tunable Constants

All thresholds are defined at the top of `Main.py` and documented inline:

| Constant | Default | Effect |
|---|---|---|
| `CPR_BPM_MIN / MAX` | 40 / 200 | CWT frequency search band |
| `CPR_TARGET_BPM` | 110 | Ideal BPM for candidate ranking |
| `MIN_PERIODICITY` | 0.25 | Minimum CWT periodicity score to be considered rhythmic |
| `MIN_BPM` | 70 | Physiological plausibility floor |
| `MAX_WRIST_RATIO` | 1.5 | Max wrist spread relative to shoulder width |
| `MIN_ELBOW_ANGLE` | 130° | Arms must be near-straight (CPR form) |
| `MIN_FACE_WRIST` | 1.0 | Face–wrist clearance (rejects bending/object-handling) |
| `KPT_CONF_THRESH` | 0.4 | YOLO keypoint confidence cutoff |
| `REQUIRED_STABLE_FRAMES` | 15 | Frames before first BPM output is trusted |
| `MAX_GRACE_FRAMES` | 7 | Frames to hold last BPM after brief occlusion |
| `SWITCH_COOLDOWN` | 30 | Frame cooldown between performer identity switches |

---

## Known Limitations

- **90° side profile** — self-occlusion of the far arm causes the system to safely output `null` rather than guess
- **Extreme motion blur / low light** — YOLO confidence drops below threshold; system halts rather than emitting a false rate
- **Camera roll** — severe diagonal tilt dampens the 1D vertical signal; keep the device roughly upright
- **Compression depth** — 2D projection cannot measure absolute depth (5–6 cm clinical target); frequency only

---

## Thesis & Citation

The full thesis PDF is available in this repository. If you use this code or methodology, please cite:

```
Hamarik, Nikolas Ján. Visual Inspection of Chest Compressions During Cardiopulmonary
Resuscitation. Bachelor's thesis, Charles University, Faculty of Mathematics and Physics,
Department of Software and Computer Science Education, Prague, 2026.
Supervisor: Ing. Adam Novozámský, Ph.D.
```

---

## Acknowledgements

- **Martin Dybal** — Záchranka application team, for proposing the collaboration and providing integration requirements
- **Emergency Medical Services, Hradec Králové Region** — for providing test video material and operational feedback
- **Ing. Adam Novozámský, Ph.D.** — thesis supervisor

---

## License

This repository is released for academic and research purposes. See `LICENSE` for details.