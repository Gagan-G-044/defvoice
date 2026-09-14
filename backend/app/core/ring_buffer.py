"""
Fixed-capacity sliding audio window.

Contract: you push arbitrary-length int16 PCM (whatever size the RTP depacketiser
hands you -- aiortc gives 20ms frames, so 320 samples) and it yields a full
WINDOW_SAMPLES float32 window every HOP_SAMPLES. Nothing else in the pipeline
needs to care about frame sizes.
"""
from __future__ import annotations

from typing import Iterator, Optional

import numpy as np

from .config import HOP_SAMPLES, SAMPLE_RATE, WINDOW_SAMPLES


class SlidingWindowBuffer:
    def __init__(
        self,
        window_samples: int = WINDOW_SAMPLES,
        hop_samples: int = HOP_SAMPLES,
    ) -> None:
        if hop_samples <= 0 or hop_samples > window_samples:
            raise ValueError("hop must be in (0, window]")
        self.window_samples = window_samples
        self.hop_samples = hop_samples
        # float32 in [-1, 1]; models want float, and converting once here keeps
        # the hot path allocation-free downstream.
        self._buf = np.zeros(window_samples, dtype=np.float32)
        self._filled = 0          # valid samples currently in _buf
        self._since_last_emit = 0  # samples pushed since we last yielded
        self.total_samples = 0     # lifetime counter, used for timestamps

    # ------------------------------------------------------------------
    @property
    def is_primed(self) -> bool:
        """True once we have seen at least one full window of audio."""
        return self._filled >= self.window_samples

    def reset(self) -> None:
        self._buf.fill(0.0)
        self._filled = 0
        self._since_last_emit = 0

    # ------------------------------------------------------------------
    def push(self, pcm: np.ndarray) -> Iterator[np.ndarray]:
        """Append samples; yield a copy of the window each time a hop elapses.

        Accepts int16 (raw PCM) or float32 (already normalised).
        """
        if pcm.dtype == np.int16:
            chunk = pcm.astype(np.float32) / 32768.0
        elif pcm.dtype == np.float32:
            chunk = pcm
        else:
            chunk = pcm.astype(np.float32)

        if chunk.ndim > 1:            # downmix any accidental stereo
            chunk = chunk.mean(axis=1, dtype=np.float32)

        n = chunk.size
        # An empty chunk is not hypothetical: AudioResampler can hand back a
        # zero-sample plane at the start of a stream. Without this guard the
        # else-branch below evaluates self._buf[:-0], which numpy reads as an
        # EMPTY slice rather than the whole array, and the assignment raises
        # inside the RTC receive coroutine -- i.e. the call drops on connect.
        if n == 0:
            return
        self.total_samples += n

        # A chunk longer than the window can only ever leave its own tail.
        if n >= self.window_samples:
            self._buf[:] = chunk[-self.window_samples :]
            self._filled = self.window_samples
        else:
            self._buf[:-n] = self._buf[n:]
            self._buf[-n:] = chunk
            self._filled = min(self.window_samples, self._filled + n)

        self._since_last_emit += n
        while self._since_last_emit >= self.hop_samples:
            self._since_last_emit -= self.hop_samples
            if self.is_primed:
                yield self._buf.copy()

    # ------------------------------------------------------------------
    def peek(self) -> Optional[np.ndarray]:
        return self._buf.copy() if self.is_primed else None


class RollingAsrBuffer:
    """Longer, lower-cadence buffer for the Whisper keyword branch.

    Whisper-tiny on a 2s window hallucinates. Give it 8s of context and only ask
    every 3s, on a separate task, so it never blocks the CM/ASV hot path.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE, seconds: float = 8.0) -> None:
        self.sample_rate = sample_rate
        self.capacity = int(sample_rate * seconds)
        self._buf = np.zeros(self.capacity, dtype=np.float32)
        self._filled = 0

    def push(self, pcm: np.ndarray) -> None:
        chunk = pcm.astype(np.float32) / 32768.0 if pcm.dtype == np.int16 else pcm
        n = chunk.size
        if n == 0:                     # see the note in the window buffer's push
            return
        if n >= self.capacity:
            self._buf[:] = chunk[-self.capacity :]
            self._filled = self.capacity
            return
        self._buf[:-n] = self._buf[n:]
        self._buf[-n:] = chunk
        self._filled = min(self.capacity, self._filled + n)

    def snapshot(self) -> np.ndarray:
        """Only the valid region, so early calls are not padded with silence."""
        return self._buf[self.capacity - self._filled :].copy()

    @property
    def seconds_held(self) -> float:
        return self._filled / float(self.sample_rate)
