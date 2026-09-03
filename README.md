# SmartErgoVision

Real-time ergonomic-risk foundation using confidence-aware pose fusion, Kalman smoothing, joint-angle features, and sequence-ready feature windows. Heavy pose models are optional adapters; the tested core is model-independent.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
python -m smartergo.demo
pytest
```

## Architecture

1. Receive MediaPipe and/or YOLOv8-Pose keypoints.
2. Fuse corresponding points using detector confidence.
3. Smooth each coordinate with a constant-velocity Kalman filter.
4. Compute neck, trunk, elbow, and knee angles.
5. Assemble fixed-length sequences for a BiLSTM-attention classifier.

The repository provides the preprocessing and deterministic risk-rule baseline. A trained BiLSTM checkpoint is not included because the original labeled training data and checkpoint were not recoverable.
