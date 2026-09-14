"""
Sliding-window buffer tests.

The buffer is driven by aiortc's 20ms audio frames (320 samples at 16kHz), which
divide neither the 2.0s window nor the 0.5s hop evenly in the general case. These
tests pin the emit cadence so a frame-size change upstream cannot silently halve
the telemetry rate.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.core import config as C
from app.core.ring_buffer import RollingAsrBuffer, SlidingWindowBuffer


FRAME = 320   # 20ms at 16kHz, what aiortc hands us


def test_no_emit_before_first_full_window():
    buf = SlidingWindowBuffer()
    emitted = 0
    pushed = 0
    while pushed < C.WINDOW_SAMPLES - FRAME:
        emitted += len(list(buf.push(np.zeros(FRAME, dtype=np.int16))))
        pushed += FRAME
    assert emitted == 0, "must not score a partially-filled window"
    assert not buf.is_primed


def test_emit_cadence_matches_hop():
    buf = SlidingWindowBuffer()
    seconds = 10.0
    total = int(seconds * C.SAMPLE_RATE)
    emitted = 0
    for _ in range(total // FRAME):
        emitted += len(list(buf.push(np.zeros(FRAME, dtype=np.int16))))

    hops = total // C.HOP_SAMPLES
    priming_hops = C.WINDOW_SAMPLES // C.HOP_SAMPLES - 1   # suppressed while filling
    assert emitted == hops - priming_hops
    # 10s of audio at a 0.5s hop, minus the 2s of priming => 17 verdicts.
    assert emitted == 17


def test_window_holds_the_most_recent_samples():
    buf = SlidingWindowBuffer()
    # Modulo keeps every value inside int16 range; a bare arange would wrap
    # silently and make this test pass for the wrong reason.
    ramp = (np.arange(C.WINDOW_SAMPLES * 2) % 4096).astype(np.int16)
    windows = []
    for i in range(0, ramp.size, FRAME):
        windows.extend(buf.push(ramp[i : i + FRAME]))
    assert windows
    last = windows[-1]
    assert last.shape == (C.WINDOW_SAMPLES,)
    consumed = (ramp.size // FRAME) * FRAME
    expected = ramp[consumed - C.WINDOW_SAMPLES : consumed].astype(np.float32) / 32768.0
    np.testing.assert_allclose(last, expected, rtol=0, atol=1e-6)


def test_int16_scaling_and_dtype():
    buf = SlidingWindowBuffer()
    loud = np.full(C.WINDOW_SAMPLES, 32767, dtype=np.int16)
    out = list(buf.push(loud))
    assert out, "a full-window push should emit"
    w = out[-1]
    assert w.dtype == np.float32
    assert 0.99 < float(w.max()) <= 1.0


def test_chunk_larger_than_window_keeps_the_tail():
    buf = SlidingWindowBuffer()
    big = (np.arange(C.WINDOW_SAMPLES * 3) % 4096).astype(np.int16)
    out = list(buf.push(big))
    assert out
    expected = big[-C.WINDOW_SAMPLES :].astype(np.float32) / 32768.0
    np.testing.assert_allclose(out[-1], expected, rtol=0, atol=1e-6)


def test_reset_reprimes():
    buf = SlidingWindowBuffer()
    list(buf.push(np.zeros(C.WINDOW_SAMPLES, dtype=np.int16)))
    assert buf.is_primed
    buf.reset()
    assert not buf.is_primed
    assert buf.peek() is None


@pytest.mark.parametrize("frame", [160, 320, 480, 960])
def test_arbitrary_frame_sizes_still_emit(frame):
    """Opus at other ptimes, or a resampler that batches differently."""
    buf = SlidingWindowBuffer()
    emitted = 0
    for _ in range((6 * C.SAMPLE_RATE) // frame):
        emitted += len(list(buf.push(np.zeros(frame, dtype=np.int16))))
    assert emitted >= 7, f"frame={frame} produced only {emitted} verdicts in ~6s"


# ---------------------------------------------------------------------------
def test_asr_buffer_reports_only_real_audio():
    buf = RollingAsrBuffer(C.SAMPLE_RATE, 8.0)
    buf.push(np.ones(C.SAMPLE_RATE, dtype=np.int16))       # 1s
    assert buf.seconds_held == pytest.approx(1.0, abs=1e-6)
    snap = buf.snapshot()
    assert snap.size == C.SAMPLE_RATE, "must not pad Whisper with leading silence"


def test_asr_buffer_caps_at_capacity():
    buf = RollingAsrBuffer(C.SAMPLE_RATE, 8.0)
    buf.push(np.zeros(C.SAMPLE_RATE * 20, dtype=np.int16))
    assert buf.seconds_held == pytest.approx(8.0, abs=1e-6)
    assert buf.snapshot().size == C.SAMPLE_RATE * 8


# ---------------------------------------------------------------------------
def test_empty_push_is_a_no_op():
    """AudioResampler can hand back a zero-sample plane at stream start.

    Regression test. Without an explicit guard the shift arithmetic evaluates
    self._buf[:-0], which numpy reads as an empty slice rather than the whole
    array, and the assignment raises ValueError inside the RTC receive coroutine
    -- so the call dies on connect and the traceback points at numpy.
    """
    buf = SlidingWindowBuffer()
    list(buf.push(np.zeros(C.WINDOW_SAMPLES, dtype=np.int16)))
    before = buf.peek()
    assert list(buf.push(np.zeros(0, dtype=np.int16))) == []
    np.testing.assert_array_equal(buf.peek(), before)

    asr = RollingAsrBuffer(C.SAMPLE_RATE, 8.0)
    asr.push(np.ones(C.SAMPLE_RATE, dtype=np.int16))
    asr.push(np.zeros(0, dtype=np.int16))
    assert asr.seconds_held == pytest.approx(1.0, abs=1e-6)
