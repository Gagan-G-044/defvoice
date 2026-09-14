#!/usr/bin/env python3
"""
Attacker client -- the demo path that cannot fail.

Joins a session as the *caller* over WebRTC and streams a WAV file (your cloned
audio) as its microphone. Run it on the laptop, a second laptop, or anywhere on
the LAN.

    python tools/attacker_cli.py --session call_001 --wav clones/cfo_wire_8lakh.wav

Why you want this even though you have two phones
-------------------------------------------------
Injecting a file into flutter_webrtc's outgoing audio track is not supported on
Android -- there is no custom audio source API. So Phone A's "attack mode" has to
play the clone out of its loudspeaker and let its own microphone pick it up. That
is honest (it is literally how a real attacker with a laptop and a phone works)
and it is a harder detection problem, but it also means room acoustics, AEC, and
speaker volume are now variables in your live demo.

Use both. Lead with Phone A's acoustic playback because it is the more convincing
story. Keep this CLI warm in a second terminal as the deterministic fallback --
bit-exact audio, no room, no AEC. Framing for judges: a compromised SIP endpoint
streaming synthetic audio directly, which is also a real attack.

Present it honestly either way: this is the attacker's tooling, not the
detector's. It sends audio and tells the backend nothing about what it sent.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys

from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
log = logging.getLogger("attacker")


def build_player(path: str, loop: bool) -> MediaPlayer:
    """MediaPlayer's `loop` kwarg is not in every aiortc release."""
    try:
        return MediaPlayer(path, loop=loop)
    except TypeError:
        if loop:
            log.warning("this aiortc has no loop= support; playing once")
        return MediaPlayer(path)


async def run(args: argparse.Namespace) -> int:
    import websockets

    if not os.path.exists(args.wav):
        log.error("no such file: %s", args.wav)
        return 1

    url = (
        f"ws://{args.host}:{args.port}/ws/signal/{args.session}"
        f"?role=caller&token={args.token}"
    )
    player = build_player(args.wav, loop=args.loop)
    if player.audio is None:
        log.error("%s has no audio stream", args.wav)
        return 1

    pc = RTCPeerConnection()
    pc.addTrack(player.audio)

    @pc.on("connectionstatechange")
    async def _on_state() -> None:
        log.info("pc state: %s", pc.connectionState)

    try:
        async with websockets.connect(url, max_size=4 * 1024 * 1024) as ws:
            await pc.setLocalDescription(await pc.createOffer())
            await ws.send(
                json.dumps({"type": "offer", "sdp": pc.localDescription.sdp})
            )
            await ws.send(
                json.dumps({"type": "attack_mode", "mode": f"cli:{os.path.basename(args.wav)}"})
            )

            raw = await asyncio.wait_for(ws.recv(), timeout=15)
            msg = json.loads(raw)
            if msg.get("type") == "error":
                log.error("backend: %s", msg.get("message"))
                return 1
            await pc.setRemoteDescription(
                RTCSessionDescription(sdp=msg["sdp"], type="answer")
            )
            if msg.get("degraded"):
                log.warning("backend is in DEGRADED (stub CM) mode -- not demo ready")

            log.info("streaming %s for %.0fs", args.wav, args.duration)
            await asyncio.sleep(args.duration)
            await ws.send(json.dumps({"type": "bye"}))
    except asyncio.TimeoutError:
        log.error("no answer from backend -- check host/port/token and that "
                  "the laptop firewall allows inbound %s", args.port)
        return 1
    finally:
        await pc.close()
        try:
            player.audio.stop()
        except Exception:  # noqa: BLE001
            pass
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True, help="cloned (or genuine) audio to send")
    ap.add_argument("--session", default="call_001")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--token", default=os.environ.get("DEFVOICE_TOKEN", "sih26104-change-me"))
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--loop", action="store_true", help="repeat the clip")
    return asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
