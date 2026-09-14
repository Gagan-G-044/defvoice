"""
Telemetry wire-contract tests.

These exist because of a bug that cost nothing to make and everything to find:
`_payload` emitted `"type": "telemetry"` while both consumers -- Phone B
(`telemetry_service.dart`, `case 'risk'`) and the browser dashboard
(`index.html`, `m.type === "risk"`) -- dispatch on `"risk"`. Nothing crashed. The
socket stayed open, frames arrived at 2 Hz, `/health` was green, and the gauge
simply never moved. That failure mode is indistinguishable from "the model does
not work", which is the worst possible thing to be debugging on stage.

So: the field names in this file are a contract with two clients that pytest
cannot see. If you rename one here, grep the Dart and the HTML in the same commit.

Run:  cd backend && python -m pytest tests/ -v
"""
from __future__ import annotations

from app.engine.pipeline import SessionAnalyzer
from app.engine.risk_aggregator import RiskAggregator, WindowEvidence


# Keys read by mobile/lib/models/telemetry.dart RiskFrame.fromJson and by
# dashboard/index.html onRisk(). Parsing on both sides is forgiving about missing
# values, which is exactly why a silent rename would not raise anywhere.
REQUIRED_KEYS = {
    "type",
    "risk",
    "authenticity_pct",
    "is_ai_clone",
    "clone_confidence_pct",
    "level",
    "p_synthetic",
    "components",
    "reasons",
    "transcript",
    "inference_ms",
    "dropped_windows",
    "degraded",
    "step_up_required",
}


class _FakeContext:
    last_text = "approve the transfer"


class _FakeEngines:
    """Enough surface for _payload. Deliberately does not load any model.

    SessionAnalyzer.__init__ stores the bundle without touching it, so this stays
    a unit test: no torch, no speechbrain, no 2 GB download in CI.
    """

    degraded = False
    context = _FakeContext()

    def status(self) -> dict:
        return {"cm": "stub", "asv": "stub", "vad": "stub"}


def _analyzer() -> SessionAnalyzer:
    return SessionAnalyzer("call_test", _FakeEngines(), lambda _p: None)


def _verdict(p: float):
    agg = RiskAggregator()
    v = None
    for _ in range(6):
        v = agg.update(
            WindowEvidence(
                p_synthetic=p, cosine_similarity=0.78, enrolled=True,
                voiced_ratio=0.9,
            )
        )
    return v


def test_frame_type_is_risk():
    """The one assertion that would have caught the original bug."""
    payload = _analyzer()._payload(_verdict(0.05))
    assert payload["type"] == "risk", (
        "both clients switch on 'risk'; renaming this makes the gauge freeze "
        "while every health check stays green"
    )


def test_frame_carries_every_key_the_clients_read():
    payload = _analyzer()._payload(_verdict(0.95))
    missing = REQUIRED_KEYS - set(payload)
    assert not missing, f"clients read these but the backend stopped sending: {missing}"


def test_authenticity_is_the_complement_of_risk():
    """Phone B shows authenticity; the dashboard plots risk. Same number."""
    payload = _analyzer()._payload(_verdict(0.95))
    assert payload["authenticity_pct"] == 100 - payload["risk"]


def test_step_up_flag_tracks_the_level():
    calm = _analyzer()._payload(_verdict(0.05))
    assert calm["step_up_required"] is False
    assert calm["level"] == "SAFE"

    alarmed = _analyzer()._payload(_verdict(0.95))
    assert alarmed["step_up_required"] is True
    assert alarmed["level"] in ("SUSPICIOUS", "CRITICAL")


def test_types_survive_json():
    """The payload goes through json.dumps in main._broadcast.

    numpy scalars serialise fine right up until they do not, and the failure
    lands inside a fire-and-forget task where nobody sees it.
    """
    import json

    payload = _analyzer()._payload(_verdict(0.95))
    round_tripped = json.loads(json.dumps(payload))
    assert round_tripped["type"] == "risk"
    assert isinstance(round_tripped["components"], dict)
    assert isinstance(round_tripped["reasons"], list)
