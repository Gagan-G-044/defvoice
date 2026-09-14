"""
DefVoice backend -- FastAPI + aiortc middlebox + telemetry fan-out.

Endpoints
  GET  /health                        engine status, safe to hit from the phones
  WS   /ws/signal/{sid}?role=&token=  WebRTC offer/answer for caller and receiver
  WS   /ws/telemetry/{sid}?token=     risk frames pushed to Phone B and dashboard
  GET  /api/sessions                  live session list (dashboard)
  GET  /                              single-file judge dashboard

SECURITY NOTE, READ BEFORE DEMO DAY
-----------------------------------
This service carries live voice audio and has only a shared-secret token on the
WebSocket handshake. There is no TLS, no per-user auth, and no rate limiting.
That is an acceptable trade for an isolated demo network and nothing else.

On an exhibition floor, do NOT put this on venue Wi-Fi: most venue APs enable
client isolation (so the phones cannot reach the laptop at all) and anyone on the
subnet could connect to this port. Run a laptop hotspot or a travel router that
you control and set DEFVOICE_TOKEN to something unguessable.

Nothing here writes audio to disk. Keep it that way: the moment a recording of
someone's voice is persisted, this stops being a demo and starts being a system
that needs a retention policy.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Optional

from aiortc import RTCPeerConnection, RTCSessionDescription
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from .core import config as C
from .engine.pipeline import EngineBundle, SessionAnalyzer
from .rtc.middlebox import RECEIVER_WAIT_TIMEOUT, SessionRegistry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("defvoice")

app = FastAPI(title="DefVoice", version="0.1.0")

registry = SessionRegistry()
engines: Optional[EngineBundle] = None


# ---------------------------------------------------------------------------
@app.on_event("startup")
async def _startup() -> None:
    global engines
    loop = asyncio.get_running_loop()
    # Model loading is slow and blocking; keep the event loop responsive so the
    # phones' /health polls succeed while weights are still coming up.
    engines = await loop.run_in_executor(None, EngineBundle)
    if engines.degraded:
        log.warning("STARTED IN DEGRADED MODE -- CM scores are not meaningful")


@app.on_event("shutdown")
async def _shutdown() -> None:
    await registry.close_all()


def _check_token(token: str) -> None:
    if token != C.AUTH_TOKEN:
        raise HTTPException(status_code=403, detail="bad token")


# ---------------------------------------------------------------------------
@app.get("/health")
async def health() -> JSONResponse:
    """Deliberately unauthenticated.

    This is the phone's "can I reach the laptop at all?" probe, and it has to
    answer before the operator has necessarily typed a correct token -- otherwise
    "the laptop is unreachable" and "your token is wrong" produce the same red
    error, which is exactly the confusion that eats fifteen minutes on demo day.

    It therefore returns capability and tuning information only. Live session ids
    are NOT listed here: the token is the only thing gating /ws/signal, so there
    is no reason to hand out the other half of the pair for free.
    """
    return JSONResponse(
        {
            "ok": True,
            "engines": engines.status() if engines else {"loading": True},
            "degraded": engines.degraded if engines else True,
            "config": {
                "window_sec": C.WINDOW_SEC,
                "hop_sec": C.HOP_SEC,
                "alpha_rise": C.EWMA_ALPHA_RISE,
                "alpha_fall": C.EWMA_ALPHA_FALL,
                "promote_dwell": C.PROMOTE_DWELL,
                "demote_dwell": C.DEMOTE_DWELL,
                "thresholds": [
                    C.THRESHOLD_CAUTION,
                    C.THRESHOLD_SUSPICIOUS,
                    C.THRESHOLD_CRITICAL,
                ],
            },
            "session_count": len(registry.all()),
        }
    )


@app.get("/api/sessions")
async def sessions(token: str = Query("")) -> JSONResponse:
    _check_token(token)
    return JSONResponse(
        [
            {
                "session_id": s.session_id,
                "caller_connected": s.caller_pc is not None,
                "receiver_connected": s.receiver_pc is not None,
                "attack_mode": s.attack_mode,
                "telemetry_clients": len(s.telemetry_subscribers),
            }
            for s in registry.all()
        ]
    )


DASHBOARD_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..",
        "dashboard",
        "index.html",
    )
)


@app.get("/")
@app.get("/dashboard")
@app.get("/dashboard/")
@app.get("/dashboard/index.html")
async def dashboard() -> FileResponse:
    return FileResponse(DASHBOARD_PATH)


# ---------------------------------------------------------------------------
def _broadcast(session, payload: dict) -> None:
    """Fire-and-forget telemetry fan-out. Never awaited from the media path."""
    dead = []
    for ws in list(session.telemetry_subscribers):
        try:
            asyncio.ensure_future(ws.send_text(json.dumps(payload)))
            if payload.get("type") == "telemetry":
                risk_payload = dict(payload, type="risk")
                asyncio.ensure_future(ws.send_text(json.dumps(risk_payload)))
        except Exception:  # noqa: BLE001
            dead.append(ws)
    for ws in dead:
        session.telemetry_subscribers.discard(ws)


# ---------------------------------------------------------------------------
@app.websocket("/ws/signal/{session_id}")
async def signal(
    websocket: WebSocket,
    session_id: str,
    role: str = Query(...),
    token: str = Query(""),
) -> None:
    if token != C.AUTH_TOKEN:
        await websocket.close(code=4403)
        return
    if role not in ("caller", "receiver"):
        await websocket.close(code=4400)
        return

    await websocket.accept()
    session = registry.get_or_create(session_id)
    pc = RTCPeerConnection()
    log.info("[%s] %s signaling connected", session_id, role)

    if role == "caller":
        session.caller_pc = pc
        # Answer the caller's sendrecv offer with a return path that is silent
        # until Phone B joins. Avoids renegotiating the caller later.
        pc.addTrack(session.return_track)

        @pc.on("track")
        def _on_track(track):  # noqa: ANN001
            if track.kind != "audio":
                return
            log.info("[%s] caller audio track received", session_id)
            session.inbound_track = track
            analyzer = SessionAnalyzer(
                session_id,
                engines,
                on_telemetry=lambda p: _broadcast(session, p),
            )
            session.attach_analyzer(analyzer)
            session.start_analysis()
            session.track_ready.set()

            @track.on("ended")
            async def _ended() -> None:
                log.info("[%s] caller track ended", session_id)
                session.track_ready.clear()
    else:
        session.receiver_pc = pc

        @pc.on("track")
        def _on_receiver_track(track):  # noqa: ANN001
            if track.kind == "audio":
                session.attach_receiver_audio(track)

    @pc.on("connectionstatechange")
    async def _on_state() -> None:
        log.info("[%s] %s pc state -> %s", session_id, role, pc.connectionState)
        if pc.connectionState in ("failed", "closed"):
            await pc.close()

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            mtype = msg.get("type")

            if mtype == "offer":
                await pc.setRemoteDescription(
                    RTCSessionDescription(sdp=msg["sdp"], type="offer")
                )
                if role == "receiver":
                    try:
                        await asyncio.wait_for(
                            session.track_ready.wait(), timeout=RECEIVER_WAIT_TIMEOUT
                        )
                    except asyncio.TimeoutError:
                        await websocket.send_text(
                            json.dumps(
                                {"type": "error", "message": "caller never connected"}
                            )
                        )
                        continue
                    pc.addTrack(session.outbound_track_for_receiver())

                answer = await pc.createAnswer()
                # aiortc completes ICE gathering inside setLocalDescription, so the
                # SDP we send already carries host candidates. No trickle, no STUN,
                # no TURN -- everything is on one LAN.
                await pc.setLocalDescription(answer)
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "answer",
                            "sdp": pc.localDescription.sdp,
                            "degraded": engines.degraded if engines else True,
                        }
                    )
                )
                log.info("[%s] answered %s", session_id, role)

            elif mtype == "attack_mode":
                # Informational only. The backend does NOT use this to decide
                # anything -- if it did, the demo would be rigged.
                session.attack_mode = str(msg.get("mode", "unknown"))
                _broadcast(
                    session,
                    {
                        "type": "attack_mode",
                        "session_id": session_id,
                        "mode": session.attack_mode,
                    },
                )

            elif mtype == "bye":
                break

    except WebSocketDisconnect:
        log.info("[%s] %s signaling disconnected", session_id, role)
    except Exception as exc:  # noqa: BLE001
        log.exception("[%s] signaling error: %s", session_id, exc)
    finally:
        if role == "caller":
            await registry.drop(session_id)
        else:
            session.return_track.set_source(None)
            try:
                await pc.close()
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
async def _handle_telemetry_ws(
    websocket: WebSocket, session_id: str, token: Optional[str] = None
) -> None:
    """Handle telemetry subscriber connections.

    Keeps the socket open in an idle/waiting state even if the audio session
    has not started yet, rather than disconnecting with a timeout or session-not-found error.
    """
    if token and token != C.AUTH_TOKEN:
        await websocket.close(code=4403)
        return

    await websocket.accept()
    session = registry.get_or_create(session_id)
    session.telemetry_subscribers.add(websocket)
    log.info("[%s] telemetry subscriber connected (idle/waiting state)", session_id)

    await websocket.send_text(
        json.dumps(
            {
                "type": "hello",
                "session_id": session_id,
                "status": "waiting_for_call",
                "state": "idle",
                "call_active": session.inbound_track is not None,
                "engines": engines.status() if engines else {},
                "degraded": engines.degraded if engines else True,
            }
        )
    )

    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                try:
                    await websocket.send_text("pong")
                except Exception:
                    break
            else:
                try:
                    payload = json.loads(data)
                    if isinstance(payload, dict) and payload.get("type") in ("risk", "telemetry"):
                        _broadcast(session, payload)
                except Exception:
                    pass
    except WebSocketDisconnect:
        log.info("[%s] telemetry subscriber disconnected", session_id)
    except Exception as exc:  # noqa: BLE001
        log.debug("[%s] telemetry socket ended: %s", session_id, exc)
    finally:
        session.telemetry_subscribers.discard(websocket)


@app.websocket("/ws/telemetry")
@app.websocket("/ws/telemetry/")
async def telemetry_default(
    websocket: WebSocket,
    session_id: Optional[str] = Query(None),
    session: Optional[str] = Query(None),
    sid: Optional[str] = Query(None),
    token: Optional[str] = Query(None),
) -> None:
    target_sid = session_id or session or sid or "call_001"
    await _handle_telemetry_ws(websocket, target_sid, token)


@app.websocket("/ws/telemetry/{session_id}")
async def telemetry_by_id(
    websocket: WebSocket,
    session_id: str,
    token: Optional[str] = Query(None),
) -> None:
    await _handle_telemetry_ws(websocket, session_id or "call_001", token)


@app.websocket("/ws/{session_id}/receiver")
async def ws_receiver_alias(
    websocket: WebSocket,
    session_id: str,
    token: Optional[str] = Query(None),
) -> None:
    await _handle_telemetry_ws(websocket, session_id or "call_001", token)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # `python -m app.main` from backend/ honours DEFVOICE_HOST and DEFVOICE_PORT.
    # The uvicorn CLI in the README does not -- it takes its own --host/--port and
    # never reads config.py -- so if you set those env vars, start it this way or
    # they will silently do nothing.
    import uvicorn

    log.info("binding %s:%d (from DEFVOICE_HOST/DEFVOICE_PORT)", C.HOST, C.PORT)
    if C.AUTH_TOKEN == "sih26104-change-me":
        log.warning("DEFVOICE_TOKEN is still the default. Set it before a demo.")
    uvicorn.run(app, host=C.HOST, port=C.PORT, log_level="info")
