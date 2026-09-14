"""
Countermeasure (CM) engine -- "is this speech machine-generated?"

Three interchangeable backends, chosen automatically at startup:

  onnx    : exported XLS-R front-end + trained head. Fastest, what you demo.
  torch   : transformers Wav2Vec2Model (frozen) + your trained head checkpoint.
            Use while you are still iterating and have not exported yet.
  stub    : NOT A DETECTOR. See the warning on StubCm below.

POLARITY WARNING
----------------
ASVspoof codebases disagree on whether class 0 is spoof or bonafide, and getting
it backwards makes the gauge go green on every clone -- the single most common
way this demo dies on stage. Run tools/check_cm_polarity.py against one known
genuine and one known cloned file before you trust anything here.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

from ..core import config as C

log = logging.getLogger(__name__)

# Which output index means "spoofed/synthetic". Verify, do not assume.
SPOOF_CLASS_INDEX = int(os.environ.get("DEFVOICE_CM_SPOOF_INDEX", "0"))


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + float(np.exp(-x)))


class StubCm:
    """A disclosed placeholder, not a countermeasure.

    It reports a weak spectral heuristic (high-band rolloff + flatness) purely so
    the end-to-end plumbing produces plausible-shaped numbers before your trained
    weights exist. Its accuracy is close to worthless on real audio and it will
    happily be fooled by a bandlimited genuine call.

    Any session using it sets `degraded: true` in every telemetry frame. If that
    flag is set during your SIH demo, you are misrepresenting the system -- stop
    and load real weights.
    """

    name = "stub"
    degraded = True

    def score(self, window: np.ndarray) -> float:
        n = window.size
        if n < 512:
            return 0.0
        spec = np.abs(np.fft.rfft(window * np.hanning(n)))
        freqs = np.fft.rfftfreq(n, d=1.0 / C.SAMPLE_RATE)
        power = spec**2 + 1e-12

        cumulative = np.cumsum(power)
        rolloff_idx = int(np.searchsorted(cumulative, 0.95 * cumulative[-1]))
        rolloff_hz = float(freqs[min(rolloff_idx, freqs.size - 1)])

        band = (freqs >= 4000) & (freqs <= 7800)
        if not np.any(band):
            return 0.0
        b = power[band]
        geo = float(np.exp(np.mean(np.log(b))))
        arith = float(np.mean(b))
        flatness = geo / (arith + 1e-12)

        # Vocoders often leave an over-smooth, sharply-truncated high band.
        rolloff_cue = np.clip((6500.0 - rolloff_hz) / 3000.0, 0.0, 1.0)
        flatness_cue = np.clip((flatness - 0.15) / 0.5, 0.0, 1.0)
        return float(np.clip(0.55 * rolloff_cue + 0.45 * flatness_cue, 0.0, 1.0))


class OnnxCm:
    name = "onnx"
    degraded = False

    def __init__(self, path: str) -> None:
        import onnxruntime as ort

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        available = ort.get_available_providers()
        providers = [p for p in providers if p in available] or ["CPUExecutionProvider"]
        self._sess = ort.InferenceSession(path, providers=providers)
        self._input = self._sess.get_inputs()[0].name
        log.info("CM ONNX loaded from %s on %s", path, providers[0])

    def score(self, window: np.ndarray) -> float:
        out = self._sess.run(None, {self._input: window.reshape(1, -1)})[0]
        arr = np.asarray(out).reshape(-1)
        if arr.size == 1:
            return _sigmoid(float(arr[0]))
        probs = _softmax(arr.astype(np.float64))
        return float(probs[SPOOF_CLASS_INDEX])


class TorchCm:
    """Frozen SSL front-end + your trained head, run in PyTorch.

    Expects a checkpoint saved by ml/train_cm_head.py containing:
        {"head_state": ..., "encoder_name": ..., "layer": int}
    """

    name = "torch"
    degraded = False

    def __init__(self, ckpt_path: str) -> None:
        import torch
        from transformers import Wav2Vec2Model

        self._torch = torch
        ckpt = torch.load(ckpt_path, map_location="cpu")
        # Default matches ml/cache_features.py's SSL_LAYER. train_cm_head.py
        # always writes the real value, so the default only ever applies to a
        # hand-made checkpoint -- keep the two in step anyway.
        self._layer = int(ckpt.get("layer", 5))
        encoder_name = ckpt.get("encoder_name", "facebook/wav2vec2-xls-r-300m")

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._encoder = Wav2Vec2Model.from_pretrained(encoder_name).to(self._device)
        self._encoder.eval()
        for p in self._encoder.parameters():
            p.requires_grad_(False)

        from .cm_head import LinearHead  # local import to avoid a cycle

        hidden = self._encoder.config.hidden_size
        self._head = LinearHead(hidden).to(self._device)
        self._head.load_state_dict(ckpt["head_state"])
        self._head.eval()
        # Encoder only. The head stays fp32 on purpose: score() feeds it
        # feat.float(), and a half-precision head against a float input raises
        # "Input type ... and weight type ... should be the same" on every
        # window. CmEngine.score catches that and returns None, so the risk
        # aggregator holds its prior threat state -- but the gauge becomes
        # unresponsive to new audio, which is still a failure worth avoiding.
        # The head is ~0.9M params; fp32 costs nothing measurable.
        if self._device.type == "cuda":
            self._encoder.half()
        log.info("CM torch backend ready (%s, layer %d) on %s",
                 encoder_name, self._layer, self._device)

    def score(self, window: np.ndarray) -> float:
        torch = self._torch
        with torch.inference_mode():
            x = torch.from_numpy(window).float().unsqueeze(0).to(self._device)
            # Per-utterance zero-mean / unit-variance. XLS-R ships with
            # do_normalize=True in its feature extractor, and ml/cache_features.py
            # applies exactly this before caching. If these two ever disagree the
            # head sees a different input distribution than it trained on and the
            # scores become noise -- with a healthy validation EER to hide it.
            x = (x - x.mean(dim=-1, keepdim=True)) / (
                x.std(dim=-1, keepdim=True) + 1e-7
            )
            if self._device.type == "cuda":
                x = x.half()
            hs = self._encoder(x, output_hidden_states=True).hidden_states
            feat = hs[self._layer]                      # (1, T, H)
            # Training cached features as fp16 (cache_features.py line 112).
            # On CPU the encoder runs fp32, so the head sees a distribution it
            # never trained on. Round-trip through fp16 to match training.
            feat = feat.half().float()
            logits = self._head(feat)                   # (1, 2)
            probs = torch.softmax(logits, dim=-1)
            return float(probs[0, SPOOF_CLASS_INDEX].item())


class CmEngine:
    """Facade that picks the best available backend."""

    def __init__(self) -> None:
        self.backend = self._select()

    def _select(self):
        torch_ckpt = os.path.join(C.MODELS_DIR, "cm_head.pt")
        if os.path.exists(C.CM_ONNX):
            try:
                return OnnxCm(C.CM_ONNX)
            except Exception as exc:  # noqa: BLE001
                log.error("CM ONNX failed to load: %s", exc)
        if os.path.exists(torch_ckpt):
            try:
                return TorchCm(torch_ckpt)
            except Exception as exc:  # noqa: BLE001
                log.error("CM torch backend failed to load: %s", exc)
        if not C.ALLOW_STUB_ENGINES:
            raise RuntimeError(
                "No CM weights found and DEFVOICE_ALLOW_STUB=0. Refusing to start."
            )
        log.warning("=" * 70)
        log.warning("CM ENGINE RUNNING IN STUB MODE -- SCORES ARE NOT MEANINGFUL")
        log.warning("=" * 70)
        return StubCm()

    @property
    def degraded(self) -> bool:
        return self.backend.degraded

    @property
    def name(self) -> str:
        return self.backend.name

    def score(self, window: np.ndarray) -> Optional[float]:
        """Score an audio window. Returns p(synthetic) in [0.0, 1.0], or None on error."""
        try:
            return float(np.clip(self.backend.score(window), 0.0, 1.0))
        except Exception as exc:  # noqa: BLE001 - a bad window must not drop the call
            log.error("CM inference failed on window: %s", exc, exc_info=True)
            return None
