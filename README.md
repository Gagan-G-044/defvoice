# DefVoice — SIH26104

Real-time detection of voice-cloning impersonation on a live call, running
entirely on one laptop over local Wi-Fi.

Phone A calls Phone B through a laptop that sits in the media path as a WebRTC
peer. The laptop decodes the audio it is already relaying, scores every 2-second
window for synthetic-speech artefacts and speaker mismatch, and pushes a verdict
to Phone B roughly every half second.

## What this does, and what it does not

It detects cloned speech **on calls placed inside this app**, where the app owns
both ends of the media path.

It does **not** touch cellular, PSTN, WhatsApp, or the native Android dialer.
Unrooted Android exposes no API that lets a third-party app intercept two-way
call audio, and no amount of engineering changes that. Any project claiming
otherwise is either rooted, wrong, or lying. The in-app VoIP call is the honest
demonstrable form of the idea; the deployment story is an operator running this
at the IMS/SBC layer, where the media already passes through infrastructure they
control.

It is **not a universal detector**. It detects the attacks it was measured on,
over the call path it was measured over. Expect roughly double your validation
EER against a TTS system it has never seen. Say the number, name the vocoders,
and say that out loud before a judge asks.

## Architecture

```
  Phone A (caller)                  Laptop  (RTX 3050)                Phone B (receiver)
  ────────────────                  ──────────────────                ──────────────────
  flutter_webrtc  ──── Opus ────►   aiortc peer                       flutter_webrtc
  mic OR loudspeaker                     │  decode → 16 kHz mono
  playback of a clone                    │
                                    ring buffer (2 s window / 0.5 s hop)
                                         │
                                    Silero VAD ──unvoiced──► skip
                                         │
                                    ┌────┴────┬──────────────┐
                                    │         │              │
                                CM (XLS-R)  ASV (ECAPA)   ASR (whisper,
                                p(synth)    cosine vs      8 s buffer,
                                    │       anchor         own cadence)
                                    └────┬────┴──────────────┘
                                         │  fuse: max(cm, asv), CM vetoes
                                    EWMA (rise 0.50 / fall 0.15)
                                         │
                                    dwell hysteresis  SAFE→CAUTION→SUSPICIOUS→CRITICAL
                                         │
                                    ─── Opus relay ──────────────────►  audio
                                    ─── /ws/telemetry ───────────────►  risk gauge
                                         │
                                    browser dashboard at http://<laptop>:8000/
```

Fusion is `max`, not a weighted mean, and the CM can veto a high ASV score. A
competent clone is built to *maximise* speaker similarity, so "this sounds like
the enrolled executive" is not evidence of authenticity — it is what a targeted
clone looks like. That case gets a risk *bonus*, not a discount.

## Run order

### 0. Backend

```bash
cd backend
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt
python -m pytest tests/ -v
```

Run the tests before anything else. They cover the risk engine's behaviour
(genuine calls stay green, urgency alone never alarms, a single spike does not
flip the UI, pauses do not launder a clone), the attack-onset latency budget, the
ring buffer's emit cadence, and the telemetry wire contract. They need only numpy
and pytest — no model weights, no torch — so they run on a fresh clone. If they
fail, nothing downstream is worth debugging.

`test_telemetry_contract.py` earns its place: the frame `type` and its key names
are a contract with two clients pytest cannot see (Phone B and the dashboard).
Break it and every health check stays green while the gauge never moves, which
looks exactly like "the model does not work".

```bash
set DEFVOICE_TOKEN=pick-something
python -m app.main
```

`python -m app.main` reads `DEFVOICE_HOST` / `DEFVOICE_PORT` from `core/config.py`.
The `uvicorn` CLI does not — it takes its own `--host/--port` and never looks at
config.py — so if you go that route (`python -m uvicorn app.main:app --host
0.0.0.0 --port 8000`), pass them explicitly.

`GET /health` reports which engine backends actually loaded, and is deliberately
open: it has to answer "can the phone reach the laptop at all?" before anyone has
typed a correct token, or those two failures look identical. `GET /api/sessions`
does require the token.

On a fresh clone with no weights, every engine falls back to a **stub** and
`degraded: true` appears in every telemetry frame. Stub mode exists so the
plumbing runs end to end on day one; it is not a detector, and the phone shows a
banner saying so.

Once you have real weights, set `DEFVOICE_ALLOW_STUB=0` so the backend refuses
to start silently degraded instead of quietly demoing a spectral heuristic.

### 1. Phones

Two unrooted Android phones on the same Wi-Fi as the laptop. See
`mobile/README.md` for the `flutter create` bootstrap and the `minSdk = 23` edit.

```bash
cd mobile
flutter analyze
flutter run -d <device-a>
flutter run -d <device-b>
```

On both phones set the laptop's LAN IP, the port, a shared session id, and the
token from `DEFVOICE_TOKEN`, then hit **Check backend** before opening either
role. Phone A places the call first.

### 2. Dashboard

Open `http://<laptop-ip>:8000/` in a browser on the laptop. No CDN, no build
step — exhibition Wi-Fi routinely has no working internet route, so the timeline
is drawn on a raw canvas.

### 3. Models (the part that turns a skeleton into a system)

Countermeasure — see `ml/README.md` for the full pipeline:

```bash
python ml/codec_augment.py data/raw/ data/aug/ --codecs opus,g711
python ml/cache_features.py manifest.tsv data/feat/ --layer 5
python ml/train_cm_head.py data/feat/ --out cm_head.pt
copy cm_head.pt backend\models\cm_head.pt
```

Enrollment for the speaker-verification branch:

```bash
cd backend
python tools/enroll.py take1.wav take2.wav take3.wav --label cfo_priya
copy models\anchor_cfo_priya.npy models\anchor_exec.npy
```

Three to five takes of 8–15 s, recorded on one device in one setting, from
someone who consented. The script reports within-speaker consistency and warns
you if the takes disagree — a blurred anchor built from mismatched recordings
matches nothing well, and the failure then looks like a threshold problem.

### 4. Calibrate — do not skip this

```bash
python tools/calibrate_asv.py --same heldout_same/ --other other_people/
python tools/check_cm_polarity.py --genuine real.wav --spoof clone.wav
python tools/bench_latency.py
```

`ASV_TAU_ACCEPT = 0.60` and `ASV_TAU_REJECT = 0.35` in `core/config.py` are
placeholders from ECAPA tutorials measured on clean VoxCeleb. They do not survive
a 16 kHz Opus leg captured on a phone in a noisy hall. Uncalibrated, you get
either the enrolled speaker flagged as a stranger or everyone passing.

`check_cm_polarity.py` answers one question: is class 0 spoof or bonafide? Get it
backwards and the gauge goes green on every clone. This is the most common way
this kind of demo dies on stage, and it takes thirty seconds to rule out.

`bench_latency.py` answers the other one: can this laptop score a window in under
`HOP_SEC`? If it cannot, the analyzer's latest-wins policy drops windows silently
— the app still looks live, the gauge still moves, and your real detection
latency is two or three times what you are claiming.

## The demo, in order

1. Both phones on the laptop's Wi-Fi, laptop plugged into mains, dashboard open
   on the projector, everything else on the laptop closed.
2. Phone A → **Caller**, Phone B → **Receiver**. B opens first so telemetry is
   already live before the call connects; that way a telemetry fault looks
   different from a call fault.
3. **Genuine leg.** Speak normally into Phone A. Gauge sits high, level SAFE.
   Say the urgent-payment sentence in your own voice — it still stays green,
   because context alone never alarms. That is the false-positive story, and
   showing it first is what makes the next part credible.
4. **Attack.** On Phone A, tap a clone clip. It plays out of the loudspeaker and
   is re-captured by Phone A's own mic, which is why Phone A disables echo
   cancellation — with AEC on, WebRTC cancels the attack into silence.
5. Phone B escalates to SUSPICIOUS then CRITICAL within about two seconds, buzzes
   once, and shows the step-up prompt. The prompt recommends out-of-band
   verification rather than "hang up", because a false positive that tells someone
   to hang up on their actual CFO is a product failure.
6. Point at the dashboard: the thin grey trace is raw per-window p(synthetic), the
   indigo trace is the smoothed risk. The gap between them is the hysteresis
   doing its job — that is why one bad window does not flip the UI.

Say out loud that Phone A tells the backend which clip it is playing, and that the
backend logs it but never scores on it. Somebody will suspect that anyway, and
volunteering it is worth more than surviving the question.

If the acoustic path is fighting you — noisy hall, weak loudspeaker — the
deterministic fallback injects audio straight into the pipeline over the same
socket the phone uses:

```bash
python backend/tools/attacker_cli.py --wav clone.wav --session call_001 \
    --host <laptop-ip> --loop
```

## When it breaks on demo day

**Phone cannot reach the laptop.** Venue Wi-Fi with AP client isolation blocks
phone→laptop traffic while both still show "connected". Use USB:

```bash
adb -s <device> reverse tcp:8000 tcp:8000
```

then set the laptop IP on the phone to `127.0.0.1`. Do this for both phones. Test
it at home once so you are not learning it under the lights.

**Gauge never moves.** Check `/health` for `degraded: true`. Stub mode means no
weights loaded. Then check the polarity — a backwards spoof index looks exactly
like "detection is broken".

**Attack clip produces silence on Phone B.** Phone A's echo canceller is on. The
dialer screen requests `rawCapture: true`; if a device ignores it, use a wired
headset on Phone A and hold it near the speaker, or fall back to `attacker_cli`.

**Call connects, no audio.** The laptop is a peer, not a signalling server — if
the aiortc side died, both legs go quiet. Watch the backend log; restart the
backend and re-place the call rather than debugging the phones.

**Everything is 3 seconds late.** Run `bench_latency.py`. If the p95 is over
budget, raise `HOP_SEC` to 1.0, re-run `tests/test_risk_engine.py`, and quote the
new latency number instead of the old one.

## Repo map

```
backend/app/core/       config.py (every tuning constant), ring_buffer.py
backend/app/engine/     vad, cm (+cm_head), asv, context, risk_aggregator, pipeline
backend/app/rtc/        middlebox.py — the aiortc peer and PCM tap
backend/app/main.py     FastAPI: /health, /ws/signal/{id}, /ws/telemetry/{id}, dashboard at /
backend/tests/          risk engine behaviour, latency budget, ring buffer, wire contract
backend/tools/          enroll, calibrate_asv, check_cm_polarity, bench_latency, attacker_cli
ml/                     codec_augment, cache_features, train_cm_head, generate_eval_clones
mobile/lib/             Flutter caller + receiver, rtc_service, telemetry_service, risk_gauge
dashboard/index.html    single file, zero dependencies
```

## Ethics, briefly, because it is load-bearing here

Cloning a voice to test a detector is the same act as cloning a voice to commit
fraud; the only difference is consent. So: clone a teammate who agreed in writing
to this specific use, never a public figure, keep the clips out of the repo, and
keep the generated audio out of the training split — `generate_eval_clones.py`
enforces the last one with an actual check rather than a comment.

## Verification status

The test suite **has not been executed** and the Flutter client **has not been
compiled** — the machine this was written on had no working shell. Everything
here was written against the interfaces in the code and audited by reading, not
by running.

A static audit did find and fix two defects that would each have produced a
demo that looks healthy and detects nothing: a half-precision CM head fed
float32 input (every window raised, was swallowed by the error handler, and
returned p(synthetic) = 0.0 forever), and a telemetry frame tagged `"telemetry"`
that neither client dispatches on. Both now have regression tests. Assume more
of that class remains.

Start with:

```bash
cd backend && python -m pytest tests/ -v
cd mobile && flutter analyze
```

and treat the first run as debugging, not confirmation.
