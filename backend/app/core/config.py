"""
Central tuning constants for the DefVoice real-time pipeline.

Every number here is load-bearing for the live demo. The values were chosen so
that the *observable* time from attack onset to a CRITICAL verdict is ~2.0-2.5s.
tests/test_risk_engine.py::test_attack_onset_latency_budget asserts that budget,
so if you retune anything below, the test tells you what it did to the demo.

That budget is the *algorithmic* latency -- how many hops of corroboration the
state machine needs. It only translates into wall-clock latency if the laptop can
actually score a window in under HOP_SEC. Check that separately with
tools/bench_latency.py; if it cannot, windows are dropped and the real latency is
worse than this file implies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path


# ----------------------------------------------------------------------------
# Audio
# ----------------------------------------------------------------------------
SAMPLE_RATE = 16_000          # Hz, mono. All engines assume this.
SAMPLE_WIDTH = 2              # bytes (int16 PCM)

WINDOW_SEC = 2.0              # analysis window fed to CM + ASV
HOP_SEC = 0.5                 # how often we emit a verdict (75% overlap)

WINDOW_SAMPLES = int(WINDOW_SEC * SAMPLE_RATE)   # 32000
HOP_SAMPLES = int(HOP_SEC * SAMPLE_RATE)         # 8000


# ----------------------------------------------------------------------------
# Voice activity detection
# ----------------------------------------------------------------------------
VAD_FRAME_SAMPLES = 512       # Silero v4/v5 expects 512 @ 16k
VAD_SPEECH_PROB_THRESHOLD = 0.5
# Fraction of frames in a window that must be voiced before we spend GPU on it.
VAD_MIN_VOICED_RATIO = 0.35


# ----------------------------------------------------------------------------
# Temporal smoothing (asymmetric EWMA)
# ----------------------------------------------------------------------------
# Rise fast (react to an attack), fall slow (an attacker cannot launder a clone
# by inserting pauses).
#
# alpha_rise is deliberately 0.50 rather than higher. At 0.75 a single anomalous
# window pushes the smoothed score past the critical threshold on its own, and
# because alpha_fall is slow it then lingers there long enough to satisfy the
# dwell requirement -- i.e. one false-positive window produces a step-up prompt
# about a second later. At 0.50 it takes two corroborating windows to cross.
# tests/test_risk_engine.py::test_single_spike_does_not_flip_the_ui pins this.
EWMA_ALPHA_RISE = 0.50
EWMA_ALPHA_FALL = 0.15


# ----------------------------------------------------------------------------
# Threat state machine
# ----------------------------------------------------------------------------
THRESHOLD_CAUTION = 30
THRESHOLD_SUSPICIOUS = 55
THRESHOLD_CRITICAL = 75

# Consecutive hop-windows required to move up / down a state.
# 2 windows @ 0.5s hop == 1.0s of corroboration before we escalate.
PROMOTE_DWELL = 2
# 8 windows @ 0.5s hop == 4.0s of calm before we de-escalate.
DEMOTE_DWELL = 8


# ----------------------------------------------------------------------------
# Speaker verification (ECAPA-TDNN cosine similarity)
# ----------------------------------------------------------------------------
# IMPORTANT: these are placeholders from ECAPA tutorials measured on clean
# VoxCeleb audio. Cosine thresholds do NOT transfer to your phones + your codec
# chain. Run `python tools/calibrate_asv.py --same <dir> --other <dir>` on your
# own recordings and overwrite these before the demo.
ASV_TAU_ACCEPT = 0.60         # >= this: same speaker as enrolled anchor
ASV_TAU_REJECT = 0.35         # <  this: different speaker

# Risk contributions from the identity branch.
# 52 lands a wrong-but-real human in CAUTION, deliberately below the SUSPICIOUS
# step-up threshold: calling from an unenrolled phone is suspicious, not fraud.
# Add the context bonus (52 + 12 = 64) and it does escalate -- a stranger using
# financial-urgency language is worth interrupting.
ASV_RISK_DIFFERENT_SPEAKER = 52
ASV_RISK_INCONCLUSIVE = 35
ASV_RISK_NO_ENROLLMENT = 20

# Bonus when a synthetic sample *also* matches the enrolled voiceprint --
# that is a targeted clone of the protected executive, the worst case.
TARGETED_CLONE_BONUS = 10
TARGETED_CLONE_CM_FLOOR = 60

# Contextual (keyword) risk. Only applied once the acoustic branches are
# already elevated, so that an urgent-but-genuine call stays green.
CONTEXT_RISK_BONUS = 12
CONTEXT_MIN_BASE_RISK = 30


# ----------------------------------------------------------------------------
# Contextual NLP cadence
# ----------------------------------------------------------------------------
# Whisper on a 2.0s window is unreliable. Run it on a longer rolling buffer,
# less often, on its own cadence.
ASR_BUFFER_SEC = 8.0
ASR_INTERVAL_SEC = 3.0


# ----------------------------------------------------------------------------
# Networking
# ----------------------------------------------------------------------------
HOST = os.environ.get("DEFVOICE_HOST", "0.0.0.0")
PORT = int(os.environ.get("DEFVOICE_PORT", "8000"))

# Shared secret required on every WebSocket connect. Hackathon venue Wi-Fi is
# hostile; do not run this open. Override via env on demo day.
AUTH_TOKEN = os.environ.get("DEFVOICE_TOKEN", "sih26104-change-me")


# ----------------------------------------------------------------------------
# Model paths / feature flags
# ----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = Path(os.environ.get("DEFVOICE_MODELS", PROJECT_ROOT / "models")).resolve()

SILERO_VAD_ONNX = str(MODELS_DIR / "silero_vad.onnx")
CM_ONNX = str(MODELS_DIR / "cm_xlsr_head.onnx")
ASV_ANCHOR = str(MODELS_DIR / "anchor_exec.npy")

# When a model file is missing the engine falls back to a deterministic stub so
# the full pipeline still runs. Stub mode is reported in every telemetry frame
# as `degraded: true` -- never demo with that flag set.
ALLOW_STUB_ENGINES = os.environ.get("DEFVOICE_ALLOW_STUB", "1") == "1"


@dataclass
class RiskWeights:
    """Kept as a dataclass so the dashboard can hot-tune during red-teaming."""

    alpha_rise: float = EWMA_ALPHA_RISE
    alpha_fall: float = EWMA_ALPHA_FALL
    promote_dwell: int = PROMOTE_DWELL
    demote_dwell: int = DEMOTE_DWELL
    thresholds: tuple = field(
        default_factory=lambda: (
            THRESHOLD_CAUTION,
            THRESHOLD_SUSPICIOUS,
            THRESHOLD_CRITICAL,
        )
    )
