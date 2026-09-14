"""
Silero VAD (ONNX) wrapper with an energy-based fallback.

Two reasons VAD gating matters here beyond saving GPU cycles:
  1. Silence and line noise have no vocoder artifacts to find, so scoring them
     injects pure noise into the EWMA.
  2. An exhibition hall is loud. Gating on speech is what stops ambient babble
     from walking the score into a false CRITICAL in front of judges.

Download the weights (~1.8MB):
  curl -L -o backend/models/silero_vad.onnx \\
    https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

from ..core import config as C

log = logging.getLogger(__name__)


class VadEngine:
    def __init__(self, model_path: str = C.SILERO_VAD_ONNX) -> None:
        self.available = False
        self._sess = None
        self._state = None
        self._layout = None  # "v5" (single state tensor) or "v4" (h/c pair)
        self._load(model_path)

    # ------------------------------------------------------------------
    def _load(self, model_path: str) -> None:
        if not os.path.exists(model_path):
            log.warning("Silero VAD not found at %s -- using energy fallback", model_path)
            return
        try:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 1
            self._sess = ort.InferenceSession(
                model_path, sess_options=opts, providers=["CPUExecutionProvider"]
            )
            names = {i.name for i in self._sess.get_inputs()}
            # Silero v5 exposes a single fused `state`; v4 exposes `h` and `c`.
            self._layout = "v5" if "state" in names else "v4"
            self.reset()
            self.available = True
            log.info("Silero VAD loaded (%s layout)", self._layout)
        except Exception as exc:  # noqa: BLE001 - never let VAD kill the call
            log.warning("Silero VAD load failed (%s) -- using energy fallback", exc)
            self._sess = None

    def reset(self) -> None:
        if self._layout == "v5":
            self._state = np.zeros((2, 1, 128), dtype=np.float32)
            self._context = np.zeros((1, 64), dtype=np.float32)
        else:
            self._state = (
                np.zeros((2, 1, 64), dtype=np.float32),
                np.zeros((2, 1, 64), dtype=np.float32),
            )

    # ------------------------------------------------------------------
    def _frame_prob(self, frame: np.ndarray) -> float:
        if self._sess is None:
            # Fallback: RMS against a fixed floor. Crude, but it does separate
            # speech from room tone and it never crashes.
            rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)) + 1e-9)
            return float(np.clip((20 * np.log10(rms) + 50.0) / 25.0, 0.0, 1.0))

        f = frame.reshape(1, -1).astype(np.float32)
        sr = np.array(C.SAMPLE_RATE, dtype=np.int64)
        if self._layout == "v5":
            # Silero v5 expects (1, 576) at 16 kHz: 64-sample rolling
            # context prepended to the 512-sample frame.
            x = np.concatenate([self._context, f], axis=1)
            out, new_state = self._sess.run(
                None, {"input": x, "state": self._state, "sr": sr}
            )
            self._state = new_state
            self._context = x[:, -64:]
        else:
            out, h, c = self._sess.run(
                None, {"input": f, "sr": sr, "h": self._state[0], "c": self._state[1]}
            )
            self._state = (h, c)
        return float(np.array(out).reshape(-1)[0])

    # ------------------------------------------------------------------
    def voiced_ratio(self, window: np.ndarray) -> float:
        """Fraction of 512-sample frames in the window classified as speech."""
        n = C.VAD_FRAME_SAMPLES
        usable = (window.size // n) * n
        if usable == 0:
            return 0.0
        frames = window[:usable].reshape(-1, n)
        hits = 0
        for f in frames:
            if self._frame_prob(f) >= C.VAD_SPEECH_PROB_THRESHOLD:
                hits += 1
        return hits / float(frames.shape[0])

    def is_voiced(self, window: np.ndarray) -> tuple:
        ratio = self.voiced_ratio(window)
        return ratio >= C.VAD_MIN_VOICED_RATIO, ratio
