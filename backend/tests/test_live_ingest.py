"""
Integration test: feed a real clone WAV through SessionAnalyzer and verify
that the pipeline produces meaningful telemetry.

Requires:
  - A CM model in backend/models/ (cm_head.pt or cm_xlsr_head.onnx).
  - SpeechBrain's ECAPA-TDNN model (downloaded on first run).
  - Silero VAD ONNX weights in backend/models/.

If the heavy models are not available, the test is skipped so the rest of the
suite still passes on a fresh clone with ``python -m pytest tests/ -v``.

Run standalone:
    cd backend
    ../.venv/Scripts/python.exe -m pytest tests/test_live_ingest.py -v -s
"""
from __future__ import annotations

import asyncio
import os
import sys
from typing import List

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Pre-flight: skip early if the WAV or engine deps are unavailable.
# ---------------------------------------------------------------------------
_CLONE_WAV = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__), "..", "..",
        "mobile", "assets", "clones", "cfo_wire_8lakh_hi.wav",
    )
)

_MODELS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "models")
)

_HAS_WAV = os.path.isfile(_CLONE_WAV)
_HAS_CM = (
    os.path.isfile(os.path.join(_MODELS_DIR, "cm_head.pt"))
    or os.path.isfile(os.path.join(_MODELS_DIR, "cm_xlsr_head.onnx"))
)
_HAS_VAD = os.path.isfile(os.path.join(_MODELS_DIR, "silero_vad.onnx"))

_REASON_PARTS: list[str] = []
if not _HAS_WAV:
    _REASON_PARTS.append(f"clone WAV missing ({_CLONE_WAV})")
if not _HAS_CM:
    _REASON_PARTS.append("no CM weights (cm_head.pt / cm_xlsr_head.onnx)")
if not _HAS_VAD:
    _REASON_PARTS.append("no Silero VAD weights")

_SKIP_REASON = "; ".join(_REASON_PARTS) if _REASON_PARTS else ""

try:
    import soundfile  # noqa: F401 - needed to load the WAV
    _HAS_SF = True
except ImportError:
    _HAS_SF = False
    _SKIP_REASON = _SKIP_REASON or "soundfile not installed"

pytestmark = pytest.mark.skipif(
    bool(_SKIP_REASON), reason=_SKIP_REASON
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_wav_int16(path: str):
    """Load a WAV as int16 mono, matching what TappedAudioTrack produces."""
    import soundfile as sf
    audio, sr = sf.read(path, dtype="int16")
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio, sr


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def engine_bundle():
    """Load engines once for the whole module — this takes ~25 s."""
    from app.engine.pipeline import EngineBundle
    return EngineBundle()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestLiveIngest:
    """Feed a cloned-voice WAV through SessionAnalyzer and inspect telemetry."""

    def test_clone_wav_produces_telemetry(self, engine_bundle):
        """At least one telemetry frame must be emitted."""
        frames = self._run_analyzer(engine_bundle)
        assert len(frames) > 0, "no telemetry frames emitted"

    def test_telemetry_has_required_keys(self, engine_bundle):
        frames = self._run_analyzer(engine_bundle)
        required = {"type", "session_id", "risk", "level", "p_synthetic",
                     "components", "reasons", "window"}
        for f in frames:
            missing = required - f.keys()
            assert not missing, f"telemetry frame missing keys: {missing}"

    def test_vad_detects_voiced_speech(self, engine_bundle):
        """At least one window must pass the VAD gate (p_synthetic != None)."""
        frames = self._run_analyzer(engine_bundle)
        scored = [f for f in frames if f["p_synthetic"] is not None]
        assert len(scored) > 0, (
            f"VAD gated every window — no CM scores. "
            f"voiced_ratios: {[f['components'].get('voiced_ratio') for f in frames]}"
        )

    def test_risk_nonzero_for_clone(self, engine_bundle):
        """A cloned voice should produce risk > 0 on at least one frame."""
        frames = self._run_analyzer(engine_bundle)
        max_risk = max(f["risk"] for f in frames)
        assert max_risk > 0, (
            f"all risk scores are 0 — CM may be returning stub values. "
            f"p_synthetic values: {[f['p_synthetic'] for f in frames]}"
        )

    def test_p_synthetic_nonnull_at_least_once(self, engine_bundle):
        """p_synthetic must be a real float at least once."""
        frames = self._run_analyzer(engine_bundle)
        p_vals = [f["p_synthetic"] for f in frames if f["p_synthetic"] is not None]
        assert len(p_vals) > 0, "p_synthetic was null in every frame"
        assert any(isinstance(v, (int, float)) for v in p_vals)

    # ------------------------------------------------------------------
    # Runner — cached per class so each test doesn't reload audio.
    # ------------------------------------------------------------------
    _cached_frames: List[dict] | None = None

    @classmethod
    def _run_analyzer(cls, engine_bundle) -> List[dict]:
        if cls._cached_frames is not None:
            return cls._cached_frames

        from app.engine.pipeline import SessionAnalyzer

        collected: List[dict] = []

        def on_telemetry(data: dict) -> None:
            collected.append(data)

        audio, sr = _load_wav_int16(_CLONE_WAV)

        analyzer = SessionAnalyzer(
            session_id="test_live_ingest",
            engines=engine_bundle,
            on_telemetry=on_telemetry,
        )

        # Feed in 320-sample chunks (~20 ms @ 16 kHz), matching WebRTC delivery.
        chunk_size = 320

        async def _feed():
            for i in range(0, len(audio), chunk_size):
                analyzer.feed(audio[i : i + chunk_size])
                # Yield to the event loop so _drain() tasks can run.
                await asyncio.sleep(0.0005)
            # Allow pending inference to finish.
            await asyncio.sleep(4.0)

        asyncio.run(_feed())
        cls._cached_frames = collected
        return collected
"""Integration test for the live audio ingestion pipeline."""
