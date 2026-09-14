"""
The middlebox: Phone A --WebRTC--> laptop --WebRTC--> Phone B.

Why the laptop sits in the media path instead of tapping audio on the phone
-------------------------------------------------------------------------
There is no Dart-level API in `flutter_webrtc` for reading raw PCM out of a
*remote* audio track. Getting it requires patching the plugin's Android
`JavaAudioDeviceModule` wiring to install a playback samples-ready callback --
days of native debugging with no guarantee. Putting aiortc in the middle gets
decoded PCM for free from `AudioResampler`, needs no plugin fork, and happens to
match the production story you pitch to judges: a network-level RTP proxy at the
IMS / SBC layer.

Cost: the laptop is now load-bearing for call audio. If the backend dies, the
call dies. Hence the drop-on-error behaviour throughout -- a failing analyser
must never take the media relay with it.

Ordering: the caller registers first (it is placing a call). If the receiver
arrives first, its offer is held until the caller's track exists.
"""
from __future__ import annotations

import asyncio
import logging
from collections import deque
from fractions import Fraction
from typing import Deque, Dict, Optional

import numpy as np
from aiortc import MediaStreamTrack, RTCPeerConnection
from aiortc.contrib.media import MediaRelay
from av.audio.frame import AudioFrame
from av.audio.resampler import AudioResampler

from ..core import config as C

log = logging.getLogger(__name__)

RECEIVER_WAIT_TIMEOUT = 30.0


class TappedAudioTrack(MediaStreamTrack):
    """Forwards frames untouched while handing a 16k mono copy to a callback.

    The forwarded frame is the original, so the tap adds no audible degradation
    to what Phone B hears -- important, because judges will be listening.
    """

    kind = "audio"

    def __init__(self, source: MediaStreamTrack, on_pcm, session_id: str = "") -> None:
        super().__init__()
        self._source = source
        self._on_pcm = on_pcm
        self._session_id = session_id
        self._frame_count = 0
        self._resampler = AudioResampler(
            format="s16", layout="mono", rate=C.SAMPLE_RATE
        )

    async def recv(self):
        frame = await self._source.recv()
        try:
            resampled = self._resampler.resample(frame)
            # av >= 9 returns a list; older versions return a single frame.
            frames = resampled if isinstance(resampled, (list, tuple)) else [resampled]
            for rf in frames:
                if rf is None:
                    continue
                pcm = np.frombuffer(bytes(rf.planes[0]), dtype=np.int16)
                self._frame_count += 1
                if self._frame_count == 1:
                    log.info(
                        "[%s] first inbound audio frame: pts=%s samples=%d",
                        self._session_id, frame.pts, pcm.size,
                    )
                self._on_pcm(pcm)
        except Exception as exc:  # noqa: BLE001 - tap failure must not break audio
            log.error(
                "[%s] tap audio error (frame %d): %s",
                self._session_id, self._frame_count, exc, exc_info=True,
            )
        return frame


class SwitchableAudioTrack(MediaStreamTrack):
    """A track whose source can be attached after negotiation has finished.

    Solves an ordering problem in the middlebox. The caller connects first (it is
    placing the call) and its offer is sendrecv, so we must hand it a return track
    in the answer -- but the receiver has not joined yet, so there is nothing to
    send. Renegotiating later is possible and unpleasant. Instead we answer with
    this track, which emits silence until `set_source()` is called with the
    receiver's audio.

    It rewrites pts on every outgoing frame so the timestamp sequence stays
    monotonic across the silence-to-live switch; a jump there produces an audible
    glitch or a stalled stream depending on the receiver's jitter buffer.
    """

    kind = "audio"

    RATE = 48_000
    SAMPLES_PER_FRAME = 960          # 20ms

    def __init__(self) -> None:
        super().__init__()
        self._source: Optional[MediaStreamTrack] = None
        self._pts = 0
        self._resampler = AudioResampler(
            format="s16", layout="mono", rate=self.RATE
        )
        # recv() must return exactly one frame, but a single resample call can
        # emit several. Park the extras here and serve them on the next calls
        # instead of dropping them -- dropping is inaudible as a glitch but shows
        # up as the receiver->caller leg sounding chopped.
        self._pending: Deque = deque()

    def set_source(self, track: Optional[MediaStreamTrack]) -> None:
        self._source = track
        # Frames queued from the previous source are stale by the time anyone
        # switches; emitting them would splice a fragment of the old leg into
        # the new one.
        self._pending.clear()

    def _stamp(self, frame):
        frame.pts = self._pts
        frame.time_base = Fraction(1, self.RATE)
        self._pts += frame.samples
        return frame

    def _silence(self):
        frame = AudioFrame(
            format="s16", layout="mono", samples=self.SAMPLES_PER_FRAME
        )
        for plane in frame.planes:
            plane.update(bytes(plane.buffer_size))
        frame.sample_rate = self.RATE
        return self._stamp(frame)

    async def recv(self):
        if self._pending:
            return self._stamp(self._pending.popleft())
        src = self._source
        if src is None:
            await asyncio.sleep(self.SAMPLES_PER_FRAME / self.RATE)
            return self._silence()
        try:
            frame = await src.recv()
            resampled = self._resampler.resample(frame)
            frames = resampled if isinstance(resampled, (list, tuple)) else [resampled]
            for rf in frames:
                if rf is None:
                    continue
                rf.sample_rate = self.RATE
                self._pending.append(rf)
            if self._pending:
                return self._stamp(self._pending.popleft())
            return self._silence()
        except Exception as exc:  # noqa: BLE001
            log.info("return path source ended (%s); falling back to silence", exc)
            self._source = None
            await asyncio.sleep(self.SAMPLES_PER_FRAME / self.RATE)
            return self._silence()


class CallSession:
    """State for one two-party call."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.relay = MediaRelay()
        self.caller_pc: Optional[RTCPeerConnection] = None
        self.receiver_pc: Optional[RTCPeerConnection] = None
        self.inbound_track: Optional[MediaStreamTrack] = None
        self.return_track = SwitchableAudioTrack()   # receiver -> caller
        self.track_ready = asyncio.Event()
        self.analyzer = None                      # set by the app layer
        self.telemetry_subscribers: set = set()
        self._analysis_task: Optional[asyncio.Task] = None
        self.attack_mode: str = "live_mic"        # reported by Phone A, display only

    # ------------------------------------------------------------------
    def attach_receiver_audio(self, track: MediaStreamTrack) -> None:
        """Complete the return path once Phone B starts sending."""
        self.return_track.set_source(self.relay.subscribe(track, buffered=False))
        log.info("[%s] return path live (receiver -> caller)", self.session_id)


    # ------------------------------------------------------------------
    def attach_analyzer(self, analyzer) -> None:
        self.analyzer = analyzer

    def start_analysis(self) -> None:
        """Pull frames on our own task so analysis runs even before Phone B joins."""
        if self._analysis_task or self.inbound_track is None:
            return
        sub = self.relay.subscribe(self.inbound_track, buffered=False)
        tapped = TappedAudioTrack(sub, self._on_pcm, session_id=self.session_id)
        self._analysis_task = asyncio.ensure_future(self._pump(tapped))
        log.info("[%s] analysis pump started", self.session_id)

    async def _pump(self, track: TappedAudioTrack) -> None:
        count = 0
        try:
            while True:
                await track.recv()
                count += 1
                if count % 50 == 0:  # ~1s at 20ms/frame
                    log.info(
                        "[%s] pump heartbeat: %d frames forwarded",
                        self.session_id, count,
                    )
        except asyncio.CancelledError:
            log.info("[%s] analysis pump cancelled after %d frames", self.session_id, count)
            raise
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "[%s] analysis pump died after %d frames: %s",
                self.session_id, count, exc, exc_info=True,
            )

    def _on_pcm(self, pcm: np.ndarray) -> None:
        if self.analyzer is not None:
            try:
                self.analyzer.feed(pcm)
            except Exception as exc:  # noqa: BLE001
                log.exception("analyzer.feed failed: %s", exc)

    # ------------------------------------------------------------------
    def outbound_track_for_receiver(self) -> MediaStreamTrack:
        """An independent, unbuffered subscription so relay latency stays flat."""
        if self.inbound_track is None:
            raise RuntimeError("no caller track yet")
        return self.relay.subscribe(self.inbound_track, buffered=False)

    async def close_media(self) -> None:
        """Close audio relay and peer connections while keeping telemetry subscribers active."""
        if self._analysis_task:
            self._analysis_task.cancel()
            try:
                await self._analysis_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._analysis_task = None
        for pc in (self.caller_pc, self.receiver_pc):
            if pc is not None:
                try:
                    await pc.close()
                except Exception:  # noqa: BLE001
                    pass
        self.caller_pc = None
        self.receiver_pc = None
        self.inbound_track = None
        self.return_track = SwitchableAudioTrack()
        self.track_ready.clear()
        self.analyzer = None
        self.attack_mode = "live_mic"
        log.info("[%s] session media closed (retained in idle state for telemetry)", self.session_id)

    async def close(self) -> None:
        await self.close_media()
        self.telemetry_subscribers.clear()
        log.info("[%s] session closed", self.session_id)


class SessionRegistry:
    def __init__(self) -> None:
        self._sessions: Dict[str, CallSession] = {}

    def get_or_create(self, session_id: str) -> CallSession:
        if session_id not in self._sessions:
            self._sessions[session_id] = CallSession(session_id)
            log.info("[%s] session created", session_id)
        return self._sessions[session_id]

    def get(self, session_id: str) -> Optional[CallSession]:
        return self._sessions.get(session_id)

    def all(self):
        return list(self._sessions.values())

    async def drop(self, session_id: str) -> None:
        s = self._sessions.get(session_id)
        if s:
            if s.telemetry_subscribers:
                await s.close_media()
            else:
                self._sessions.pop(session_id, None)
                await s.close()

    async def close_all(self) -> None:
        for sid in list(self._sessions):
            await self.drop(sid)
