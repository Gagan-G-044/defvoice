"""
Risk engine tests.

The important one is test_attack_onset_latency_budget. The blueprint's demo script
promised a CRITICAL verdict "within 1.5 seconds", but its own constants
(2.0s window / 1.0s hop / alpha_rise=0.60 / 3 consecutive windows) could not
deliver before ~5s. On stage that gap reads as a broken product. This test pins
the real number so you can quote it honestly and so a retune cannot silently
regress it.

Run:  cd backend && python -m pytest tests/ -v
"""
from __future__ import annotations

import pytest

from app.core import config as C
from app.engine.risk_aggregator import RiskAggregator, ThreatLevel, WindowEvidence


CLEAN_P = 0.05
CLONE_P = 0.95
GOOD_COS = 0.78
BAD_COS = 0.18


def clean(**kw) -> WindowEvidence:
    base = dict(
        p_synthetic=CLEAN_P, cosine_similarity=GOOD_COS, enrolled=True, voiced_ratio=0.9
    )
    base.update(kw)
    return WindowEvidence(**base)


def settle(agg: RiskAggregator, n: int = 10) -> None:
    """Establish a steady authentic baseline, as a real call would."""
    for _ in range(n):
        agg.update(clean())


# ---------------------------------------------------------------------------
def test_genuine_enrolled_call_stays_safe():
    agg = RiskAggregator()
    settle(agg, 20)
    v = agg.update(clean())
    assert v.level is ThreatLevel.SAFE
    assert v.risk < C.THRESHOLD_CAUTION
    assert v.authenticity_pct > 70


def test_urgent_language_alone_never_alarms():
    """A genuine executive really does say "transfer 8 lakh immediately"."""
    agg = RiskAggregator()
    settle(agg, 20)
    for _ in range(10):
        v = agg.update(
            clean(context_hit=True, context_terms=["lakh", "immediately"])
        )
    assert v.level is ThreatLevel.SAFE, "keywords must not raise an alarm on their own"
    assert any("not escalating" in r for r in v.reasons)


def test_unenrolled_human_is_caution_not_critical():
    """Scenario 2: a different real person. Flag it, do not cry deepfake, and do
    not interrupt the call -- CAUTION sits below the step-up threshold."""
    agg = RiskAggregator()
    for _ in range(20):
        v = agg.update(
            WindowEvidence(
                p_synthetic=CLEAN_P,
                cosine_similarity=BAD_COS,
                enrolled=True,
                voiced_ratio=0.9,
            )
        )
    assert v.level is ThreatLevel.CAUTION
    assert v.level < ThreatLevel.SUSPICIOUS, "wrong person != fraud; do not step up"
    assert v.risk == pytest.approx(C.ASV_RISK_DIFFERENT_SPEAKER, abs=4)


def test_stranger_plus_urgency_does_escalate():
    """CAUTION + financial pressure language is worth interrupting."""
    agg = RiskAggregator()
    for _ in range(20):
        v = agg.update(
            WindowEvidence(
                p_synthetic=CLEAN_P,
                cosine_similarity=BAD_COS,
                enrolled=True,
                voiced_ratio=0.9,
                context_hit=True,
                context_terms=["otp", "immediately"],
            )
        )
    assert v.level >= ThreatLevel.SUSPICIOUS


def test_high_asv_match_cannot_suppress_the_cm():
    """The regression the linear-blend design would have shipped.

    A successful clone maximises cosine similarity by construction. If ASV is
    averaged in, that similarity *lowers* the score exactly when the attack is
    working. Assert the opposite: matching the enrolled voiceprint while sounding
    synthetic must score HIGHER than sounding synthetic as a stranger.
    """
    matched = RiskAggregator()
    stranger = RiskAggregator()
    for _ in range(12):
        vm = matched.update(
            WindowEvidence(
                p_synthetic=CLONE_P, cosine_similarity=GOOD_COS,
                enrolled=True, voiced_ratio=0.9,
            )
        )
        vs = stranger.update(
            WindowEvidence(
                p_synthetic=CLONE_P, cosine_similarity=BAD_COS,
                enrolled=True, voiced_ratio=0.9,
            )
        )
    assert vm.risk >= vs.risk
    assert vm.risk >= C.THRESHOLD_CRITICAL
    assert any("targeted clone" in r for r in vm.reasons)


def test_single_spike_does_not_flip_the_ui():
    """One anomalous window may tint the gauge amber; it must never interrupt.

    Crossing into SUSPICIOUS is what freezes the transaction on Phone B, so a
    lone false positive reaching that level means a self-inflicted failure in
    front of judges.
    """
    agg = RiskAggregator()
    settle(agg, 20)

    v = agg.update(clean(p_synthetic=0.99))       # one bad window
    assert v.level is ThreatLevel.SAFE, "a single window must not escalate at all"

    worst = v.level
    recovered_after = None
    for i in range(1, 25):
        v = agg.update(clean())
        worst = max(worst, v.level)
        if recovered_after is None and v.level is ThreatLevel.SAFE:
            recovered_after = i

    assert worst < ThreatLevel.SUSPICIOUS, (
        f"spike escalated to {worst.label}; step-up would have fired on a clean call"
    )
    assert recovered_after is not None, "never returned to SAFE"
    assert recovered_after * C.HOP_SEC <= 8.0


def test_pauses_do_not_launder_a_clone():
    """alpha_fall << alpha_rise, and gated windows hold state."""
    agg = RiskAggregator()
    for _ in range(8):
        agg.update(clean(p_synthetic=CLONE_P, cosine_similarity=BAD_COS))
    hot = agg.level
    assert hot is ThreatLevel.CRITICAL
    # Attacker inserts 2s of silence (4 gated windows at a 0.5s hop).
    for _ in range(4):
        v = agg.update(
            WindowEvidence(p_synthetic=None, cosine_similarity=None,
                           enrolled=True, voiced_ratio=0.0)
        )
    assert v.level is ThreatLevel.CRITICAL, "silence must not reset the verdict"


# ---------------------------------------------------------------------------
def test_attack_onset_latency_budget():
    """How long from the first cloned syllable to an actionable verdict?

    Models the one effect that dominates: window overlap. A window ending at
    time t after onset contains only min(t/WINDOW_SEC, 1) of attacked audio, so
    early windows are diluted with genuine speech. p_synthetic is interpolated
    over that fraction. This is a plumbing/tuning check, not a claim about model
    accuracy -- accuracy comes from ml/ evaluation on the unseen split.
    """
    agg = RiskAggregator()
    settle(agg, 20)

    t = 0.0
    t_step_up = None
    t_critical = None
    trace = []

    for _ in range(int(8.0 / C.HOP_SEC)):
        t += C.HOP_SEC
        frac = min(t / C.WINDOW_SEC, 1.0)
        p = CLEAN_P + (CLONE_P - CLEAN_P) * frac
        v = agg.update(
            WindowEvidence(
                p_synthetic=p, cosine_similarity=GOOD_COS,
                enrolled=True, voiced_ratio=0.9,
            )
        )
        trace.append((round(t, 2), v.risk, v.level.label))
        if t_step_up is None and v.level >= ThreatLevel.SUSPICIOUS:
            t_step_up = t
        if t_critical is None and v.level is ThreatLevel.CRITICAL:
            t_critical = t

    print("\n  t     risk  level")
    for row in trace[:10]:
        print(f"  {row[0]:<5} {row[1]:<5} {row[2]}")

    assert t_step_up is not None and t_critical is not None
    # These are the numbers to say out loud on stage. Quote them, not 1.5s.
    #
    # As tuned, step-up lands exactly ON 2.5s, so this bound has zero slack: any
    # increase to PROMOTE_DWELL, or any decrease to EWMA_ALPHA_RISE, fails it
    # immediately. That is the point. If it starts failing after you retune, the
    # test is not flaky -- you made the demo slower, and the fix is to quote the
    # new number rather than to relax the bound.
    assert t_step_up <= 2.5, f"step-up took {t_step_up}s"
    assert t_critical <= 3.0, f"CRITICAL took {t_critical}s"
    print(f"\n  step-up at {t_step_up}s, CRITICAL at {t_critical}s")
    print("  (algorithmic latency only -- add per-window inference time from "
          "tools/bench_latency.py for the wall-clock figure)")
