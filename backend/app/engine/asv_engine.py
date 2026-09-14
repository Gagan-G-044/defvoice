"""
Automatic Speaker Verification -- "is this the person we enrolled?"

Deliberately NOT exported to ONNX. ECAPA-TDNN is ~22M params and runs a 2s window
in single-digit milliseconds on an RTX 3050; exporting SpeechBrain's Fbank +
StatisticsPooling graph is a day of yak-shaving that buys you nothing. Spend that
day on threshold calibration instead.

Correction to a common spec error: `speechbrain/spkrec-ecapa-voxceleb` emits
**192**-dimensional embeddings, not 256.

What this branch can and cannot tell you
----------------------------------------
It cannot detect a clone. A competent clone is built to maximise exactly the
cosine similarity measured here, so a high score is not evidence of authenticity.
Its job is to catch the *cheap* attack -- an unrelated human reading a script --
and to identify when a synthetic sample is specifically targeting the protected
speaker. Fusion logic in risk_aggregator.py is written accordingly.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

from ..core import config as C

log = logging.getLogger(__name__)

EMBED_DIM = 192


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class AsvEngine:
    def __init__(self, anchor_path: str = C.ASV_ANCHOR) -> None:
        self.available = False
        self.enrolled = False
        self._model = None
        self._torch = None
        self._anchor: Optional[np.ndarray] = None
        self.anchor_label = "unenrolled"
        self._load_model()
        self.load_anchor(anchor_path)

    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        try:
            import torch

            try:  # speechbrain >= 1.0
                from speechbrain.inference.speaker import EncoderClassifier
            except ImportError:  # speechbrain 0.5.x
                from speechbrain.pretrained import EncoderClassifier

            run_opts = {"device": "cuda" if torch.cuda.is_available() else "cpu"}
            self._model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=os.path.join(C.MODELS_DIR, "ecapa"),
                run_opts=run_opts,
            )
            self._torch = torch
            self.available = True
            log.info("ECAPA-TDNN loaded on %s", run_opts["device"])
        except Exception as exc:  # noqa: BLE001
            log.warning("ASV unavailable (%s) -- identity branch disabled", exc)

    # ------------------------------------------------------------------
    def load_anchor(self, path: str) -> bool:
        if not os.path.exists(path):
            log.warning("No enrolled voiceprint at %s -- run tools/enroll.py", path)
            return False
        try:
            data = np.load(path, allow_pickle=True)
            if isinstance(data, np.ndarray) and data.dtype != object:
                vec = data.astype(np.float32).reshape(-1)
                label = os.path.splitext(os.path.basename(path))[0]
            else:  # saved as a dict via np.save(..., allow_pickle=True)
                d = data.item()
                vec = np.asarray(d["embedding"], dtype=np.float32).reshape(-1)
                label = d.get("label", "enrolled")
            if vec.size != EMBED_DIM:
                log.error("Anchor dim %d != expected %d", vec.size, EMBED_DIM)
                return False
            self._anchor = vec / (np.linalg.norm(vec) + 1e-9)
            self.anchor_label = label
            self.enrolled = True
            log.info("Loaded voiceprint '%s'", label)
            return True
        except Exception as exc:  # noqa: BLE001
            log.error("Failed to load anchor: %s", exc)
            return False

    # ------------------------------------------------------------------
    def embed(self, window: np.ndarray) -> Optional[np.ndarray]:
        if not self.available:
            return None
        torch = self._torch
        try:
            with torch.inference_mode():
                x = torch.from_numpy(window.astype(np.float32)).unsqueeze(0)
                emb = self._model.encode_batch(x)     # (1, 1, 192)
                v = emb.squeeze().detach().cpu().numpy().reshape(-1)
            return v / (np.linalg.norm(v) + 1e-9)
        except Exception as exc:  # noqa: BLE001
            log.exception("ASV embedding failed: %s", exc)
            return None

    def similarity(self, window: np.ndarray) -> Optional[float]:
        """Cosine similarity of this window against the enrolled anchor."""
        if not self.enrolled or self._anchor is None:
            return None
        v = self.embed(window)
        if v is None:
            return None
        return cosine(self._anchor, v)
