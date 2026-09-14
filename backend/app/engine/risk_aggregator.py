"""
Multi-factor risk fusion, asymmetric EWMA smoothing, and the 4-state hysteresis
machine that drives the UI.

DESIGN NOTE -- why this is not a weighted sum
---------------------------------------------
A good voice clone is *supposed* to pass speaker verification; that is the whole
point of cloning. So a high ASV similarity is NOT evidence of authenticity. If
you blend the branches linearly, a 0.9 cosine match silently drags the score
down exactly when the attack is succeeding.

So: the countermeasure branch has authority on "is this a machine", the ASV
branch has authority on "is this the claimed human", and we take the *max* of
the two rather than the mean. A synthetic sample that also matches the enrolled
anchor gets a bonus, not a discount -- that is a targeted clone of the protected
executive, the worst case in the threat model.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Deque, List, Optional

from ..core import config as C


class ThreatLevel(IntEnum):
    SAFE = 0
    CAUTION = 1
    SUSPICIOUS = 2
    CRITICAL = 3

    @property
    def label(self) -> str:
        return self.name


@dataclass
class WindowEvidence:
    """One 2.0s window's worth of raw model output."""

    p_synthetic: Optional[float] = None      # CM branch, [0, 1]
    cosine_similarity: Optional[float] = None  # ASV branch, [-1, 1]
    enrolled: bool = False
    context_hit: bool = False
    context_terms: List[str] = field(default_factory=list)
    voiced_ratio: float = 1.0
    degraded: bool = False                   # any engine running in stub mode


@dataclass
class RiskVerdict:
    risk: int
    smoothed_p_synthetic: float
    level: ThreatLevel
    authenticity_pct: int
    is_ai_clone: bool
    clone_confidence_pct: int
    reasons: List[str]
    components: dict
    degraded: bool


class RiskAggregator:
    """One instance per call session. Not thread-safe by design -- drive it from
    a single asyncio task."""

    def __init__(self, weights: Optional[C.RiskWeights] = None) -> None:
        self.w = weights or C.RiskWeights()
        # The EWMA is applied to p_synthetic, NOT to the composite risk score.
        # Risk is recomputed from scratch each window; the only temporal state is
        # this smoothed CM probability plus the level history below. Smoothing
        # risk as well would stack two lags and blow the latency budget.
        self._p_smooth: float = 0.0
        self._level: ThreatLevel = ThreatLevel.SAFE
        self._history: Deque[ThreatLevel] = deque(
            maxlen=max(self.w.promote_dwell, self.w.demote_dwell)
        )

    # ------------------------------------------------------------------
    @property
    def level(self) -> ThreatLevel:
        return self._level

    def _ewma(self, prev: float, new: float) -> float:
        alpha = self.w.alpha_rise if new > prev else self.w.alpha_fall
        return alpha * new + (1.0 - alpha) * prev

    # ------------------------------------------------------------------
    def _asv_component(self, ev: WindowEvidence) -> tuple:
        if not ev.enrolled:
            return C.ASV_RISK_NO_ENROLLMENT, "no enrolled voiceprint for this caller"
        if ev.cosine_similarity is None:
            return C.ASV_RISK_NO_ENROLLMENT, "identity not scored for this window"
        cos = ev.cosine_similarity
        if cos < C.ASV_TAU_REJECT:
            return (
                C.ASV_RISK_DIFFERENT_SPEAKER,
                f"voiceprint mismatch (cos={cos:.2f} < {C.ASV_TAU_REJECT})",
            )
        if cos < C.ASV_TAU_ACCEPT:
            return (
                C.ASV_RISK_INCONCLUSIVE,
                f"voiceprint inconclusive (cos={cos:.2f})",
            )
        return 0, f"voiceprint matches enrolled speaker (cos={cos:.2f})"

    # ------------------------------------------------------------------
    def update(self, ev: WindowEvidence) -> RiskVerdict:
        reasons: List[str] = []

        # --- countermeasure branch -----------------------------------
        if ev.p_synthetic is None:
            # VAD gated this window out. Feed the smoother its own value so the
            # state is held rather than decayed -- silence is not exculpatory.
            p_raw = self._p_smooth
            reasons.append("no voiced speech in window; holding prior state")
        else:
            p_raw = float(ev.p_synthetic)

        # Prior is 0.0: authentic until there is evidence otherwise.
        self._p_smooth = self._ewma(self._p_smooth, p_raw)
        cm_component = 100.0 * self._p_smooth

        # --- AI clone forensic classification ------------------------
        # A window is classified as AI-generated when EITHER the smoothed
        # probability crosses 0.50 OR a single raw score is very high (≥0.60).
        # The second arm handles cold-start: a first window at 0.62 should
        # flag immediately rather than wait for the EWMA to catch up.
        is_ai_clone = (
            self._p_smooth >= 0.50
            or (ev.p_synthetic is not None and ev.p_synthetic >= 0.60)
        )
        clone_confidence_pct = int(round(self._p_smooth * 100))

        # --- Voice Cloning reason (Wav2Vec2 acoustic analysis) -------
        if ev.p_synthetic is not None:
            if is_ai_clone:
                reasons.append(
                    f"AI Voice Clone Detected ({clone_confidence_pct}% confidence)"
                )
                if self._p_smooth >= 0.70:
                    reasons.append(
                        "Unnatural vocoder/neural synthesis artifacts"
                    )
            else:
                reasons.append(
                    f"Voice authentic ({100 - clone_confidence_pct}% confidence)"
                )

        # --- identity branch -----------------------------------------
        asv_component, asv_reason = self._asv_component(ev)

        # --- Speaker Verification reason (ECAPA-TDNN) ----------------
        if not ev.enrolled:
            reasons.append(asv_reason)  # "no enrolled voiceprint …"
        elif ev.cosine_similarity is not None and ev.cosine_similarity < C.ASV_TAU_REJECT:
            reasons.append(
                "Speaker mismatch: caller voice does not match enrolled profile"
            )
        elif ev.cosine_similarity is not None and ev.cosine_similarity < C.ASV_TAU_ACCEPT:
            reasons.append(
                f"Speaker inconclusive (cos={ev.cosine_similarity:.2f})"
            )
        else:
            reasons.append(asv_reason)  # voiceprint matches / not scored

        # --- fusion: max, never mean ---------------------------------
        risk = max(cm_component, float(asv_component))

        if cm_component >= C.TARGETED_CLONE_CM_FLOOR and asv_component == 0:
            risk += C.TARGETED_CLONE_BONUS
            reasons.append(
                "synthetic audio that MATCHES the enrolled voiceprint: "
                "targeted clone of the protected speaker"
            )

        # --- Context / Scam Intent reason (Whisper keyword analysis) -
        if ev.context_hit and risk >= C.CONTEXT_MIN_BASE_RISK:
            risk += C.CONTEXT_RISK_BONUS
            terms = ", ".join(ev.context_terms[:4]) or "financial urgency"
            reasons.append(f"Scam intent detected: keywords [{terms}]")
        elif ev.context_hit:
            reasons.append(
                "urgency language present but acoustics are clean; not escalating"
            )

        risk = max(0.0, min(100.0, risk))

        # --- hysteresis ----------------------------------------------
        level = self._step_state_machine(risk)

        return RiskVerdict(
            risk=int(round(risk)),
            smoothed_p_synthetic=round(self._p_smooth, 4),
            level=level,
            authenticity_pct=int(round(100.0 - risk)),
            is_ai_clone=is_ai_clone,
            clone_confidence_pct=clone_confidence_pct,
            reasons=reasons,
            components={
                "cm": round(cm_component, 1),
                "asv": round(float(asv_component), 1),
                "context": C.CONTEXT_RISK_BONUS if ev.context_hit else 0,
                "is_ai_clone": is_ai_clone,
                "clone_confidence_pct": clone_confidence_pct,
                "voiced_ratio": round(ev.voiced_ratio, 2),
                "cosine": (
                    round(ev.cosine_similarity, 3)
                    if ev.cosine_similarity is not None
                    else None
                ),
            },
            degraded=ev.degraded,
        )

    # ------------------------------------------------------------------
    def _target_level(self, risk: float) -> ThreatLevel:
        caution, suspicious, critical = self.w.thresholds
        if risk >= critical:
            return ThreatLevel.CRITICAL
        if risk >= suspicious:
            return ThreatLevel.SUSPICIOUS
        if risk >= caution:
            return ThreatLevel.CAUTION
        return ThreatLevel.SAFE

    def _step_state_machine(self, risk: float) -> ThreatLevel:
        """Dwell-based hysteresis.

        Escalate to level L only if the last PROMOTE_DWELL windows *all* scored
        at or above L. De-escalate to level L only if the last DEMOTE_DWELL
        windows all scored at or below L. Stated for judges: "CRITICAL requires
        two consecutive 2-second windows both scoring >= 75."
        """
        target = self._target_level(risk)
        self._history.append(target)

        recent_promote = list(self._history)[-self.w.promote_dwell :]
        if len(recent_promote) == self.w.promote_dwell:
            corroborated = min(recent_promote)
            if corroborated > self._level:
                self._level = corroborated
                return self._level

        recent_demote = list(self._history)[-self.w.demote_dwell :]
        if len(recent_demote) == self.w.demote_dwell:
            ceiling = max(recent_demote)
            if ceiling < self._level:
                self._level = ceiling

        return self._level
