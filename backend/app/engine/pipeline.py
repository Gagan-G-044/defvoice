"""
Per-session analysis orchestration.

Two real-time properties this module is responsible for:

1. **Never block the media loop.** aiortc's audio relay runs on the asyncio event
   loop. Model inference is blocking CPU/GPU work, so it is dispatched to a thread
   executor. If you call the engines inline, forwarded audio stutters and the call
   sounds broken -- which judges hear immediately.

2. **Latest-wins, not queue.** If inference falls behind the 0.5s hop we drop
   intermediate windows instead of building a backlog. A telemetry frame that is
   4s stale is worse than no frame.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable, Optional

import numpy as np

from ..core import config as C
from ..core.ring_buffer import RollingAsrBuffer, SlidingWindowBuffer
from .asv_engine import AsvEngine
from .cm_engine import CmEngine
from .context_engine import ContextEngine
from .risk_aggregator import RiskAggregator, ThreatLevel, WindowEvidence
from .vad_engine import VadEngine

log = logging.getLogger(__name__)


class EngineBundle:
    """Heavy models, loaded once per process and shared across sessions."""

    def __init__(self) -> None:
        t0 = time.perf_counter()
        self.vad = VadEngine()
        self.cm = CmEngine()
        self.asv = AsvEngine()
        self.context = ContextEngine()
        log.info("Engines ready in %.1fs (cm=%s vad=%s asv=%s asr=%s)",
                 time.perf_counter() - t0, self.cm.name, self.vad.available,
                 self.asv.enrolled, self.context.available)

    @property
    def degraded(self) -> bool:
        return self.cm.degraded

    def status(self) -> dict:
        return {
            "cm_backend": self.cm.name,
            "cm_degraded": self.cm.degraded,
            "vad": "silero" if self.vad.available else "energy-fallback",
            "asv": self.asv.anchor_label if self.asv.enrolled else None,
            "asr": self.context.available,
        }


class SessionAnalyzer:
    """One per call. Feed it PCM, it calls back with telemetry dicts."""

    def __init__(
        self,
        session_id: str,
        engines: EngineBundle,
        on_telemetry: Callable[[dict], None],
    ) -> None:
        self.session_id = session_id
        self.engines = engines
        self.on_telemetry = on_telemetry

        self.window_buf = SlidingWindowBuffer()
        self.asr_buf = RollingAsrBuffer(C.SAMPLE_RATE, C.ASR_BUFFER_SEC)
        self.risk = RiskAggregator()

        self._pending: Optional[np.ndarray] = None
        self._busy = False
        self._dropped = 0
        self._windows = 0
        self._context_hit = False
        self._context_terms: list = []
        self._last_asr = 0.0
        self._t_start = time.perf_counter()
        self._last_latency_ms = 0.0

    # ------------------------------------------------------------------
    def feed(self, pcm: np.ndarray) -> None:
        """Called from the media path. Cheap and non-blocking by contract."""
        self.asr_buf.push(pcm)
        for window in self.window_buf.push(pcm):
            self._pending = window          # latest-wins
            if self._busy:
                self._dropped += 1
            else:
                asyncio.ensure_future(self._drain())

    # ------------------------------------------------------------------
    async def _drain(self) -> None:
        if self._busy:
            return
        self._busy = True
        try:
            loop = asyncio.get_running_loop()
            while self._pending is not None:
                window, self._pending = self._pending, None
                t0 = time.perf_counter()
                evidence = await loop.run_in_executor(
                    None, self._score_window, window
                )
                self._last_latency_ms = (time.perf_counter() - t0) * 1000.0
                verdict = self.risk.update(evidence)
                self._windows += 1
                self.on_telemetry(self._payload(verdict))
                self._maybe_schedule_asr(loop)
        except Exception as exc:  # noqa: BLE001
            log.exception("analysis loop error: %s", exc)
        finally:
            self._busy = False

    # ------------------------------------------------------------------
    def _score_window(self, window: np.ndarray) -> WindowEvidence:
        """Runs in a worker thread. All blocking model calls live here."""
        # Defensive: the ring buffer already normalises int16 → float32, but a
        # test harness or future caller that feeds raw int16 would cause the CM
        # (torch.mean on an integer tensor) to throw silently.  Catch it here.
        if window.dtype != np.float32:
            window = window.astype(np.float32) / 32768.0

        voiced, ratio = self.engines.vad.is_voiced(window)
        if not voiced:
            return WindowEvidence(
                p_synthetic=None,
                cosine_similarity=None,
                enrolled=self.engines.asv.enrolled,
                context_hit=self._context_hit,
                context_terms=list(self._context_terms),
                voiced_ratio=ratio,
                degraded=self.engines.degraded,
            )
        p_syn = self.engines.cm.score(window)
        cos = self.engines.asv.similarity(window)
        if self._windows == 0:
            log.info(
                "[%s] first scored window: p_syn=%s cos=%s voiced=%.2f",
                self.session_id,
                f"{p_syn:.4f}" if p_syn is not None else "FAILED",
                f"{cos:.3f}" if cos is not None else "N/A", ratio,
            )
        return WindowEvidence(
            p_synthetic=p_syn,
            cosine_similarity=cos,
            enrolled=self.engines.asv.enrolled,
            context_hit=self._context_hit,
            context_terms=list(self._context_terms),
            voiced_ratio=ratio,
            degraded=self.engines.degraded,
        )

    # ------------------------------------------------------------------
    def _maybe_schedule_asr(self, loop: asyncio.AbstractEventLoop) -> None:
        now = time.perf_counter()
        if not self.engines.context.available:
            return
        if now - self._last_asr < C.ASR_INTERVAL_SEC:
            return
        if self.asr_buf.seconds_held < 3.0:
            return
        self._last_asr = now
        snapshot = self.asr_buf.snapshot()
        loop.run_in_executor(None, self._run_asr, snapshot)

    def _run_asr(self, audio: np.ndarray) -> None:
        hit, terms, text = self.engines.context.analyse(audio)
        self._context_hit = hit
        self._context_terms = terms
        if hit:
            log.info("[%s] urgency terms: %s", self.session_id, terms)

    # ------------------------------------------------------------------
    def _payload(self, verdict) -> dict:
        return {
            # "risk" is the wire contract. Both consumers switch on it:
            # telemetry_service.dart `case 'risk'` and dashboard index.html
            # `m.type === "risk"`. Rename it here and the gauge silently never
            # moves -- the socket stays open, frames keep arriving, and nothing
            # renders. Change all three or none.
            "type": "risk",
            "session_id": self.session_id,
            "t": round(time.perf_counter() - self._t_start, 2),
            "window": self._windows,
            "risk": verdict.risk,
            "authenticity_pct": verdict.authenticity_pct,
            "is_ai_clone": verdict.is_ai_clone,
            "clone_confidence_pct": verdict.clone_confidence_pct,
            "level": verdict.level.label,
            "level_ord": int(verdict.level),
            "p_synthetic": verdict.smoothed_p_synthetic,
            "components": verdict.components,
            "reasons": verdict.reasons,
            "transcript": self.engines.context.last_text[-160:],
            "context_terms": self._context_terms,
            "inference_ms": round(self._last_latency_ms, 1),
            "dropped_windows": self._dropped,
            "degraded": verdict.degraded,
            "step_up_required": verdict.level >= ThreatLevel.SUSPICIOUS,
            "engines": self.engines.status(),
        }
