"""
Codec + packet-loss augmentation.

WHY THIS MATTERS MORE THAN THE MODEL
------------------------------------
A countermeasure trained on clean 16 kHz WAVs and evaluated on clean 16 kHz WAVs
will report a flattering equal-error rate and then collapse on a real call. The
artefacts a CM keys on -- phase discontinuities, over-smooth spectral envelopes,
missing high-frequency detail -- live in exactly the band that G.711 and AMR-NB
throw away. Codec augmentation is not a nice-to-have; it is the difference
between a number in a slide and a system that works on stage.

The demo path is WebRTC/Opus, so Opus is the one that must be in the training mix.
G.711 and AMR-NB are there for the claim that this transfers to a PSTN or VoLTE
leg, which is the deployment story.

torchaudio.functional.apply_codec was removed (deprecated in 2.1, gone in 2.2),
so this shells out to ffmpeg. That is slower but version-proof, and augmentation
runs once into a cache rather than per training step.
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import subprocess
import sys
import tempfile
from typing import List, Optional

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16_000

# name -> (ffmpeg args for the encode leg, intermediate container)
CODECS = {
    # Narrowband PSTN. Brutal: 8 kHz, mu-law, everything above 3.4 kHz gone.
    "g711": (["-ar", "8000", "-ac", "1", "-c:a", "pcm_mulaw"], "wav"),
    # What the WebRTC demo actually uses. 24 kbps is a realistic mobile rate.
    "opus": (["-ar", "16000", "-ac", "1", "-c:a", "libopus", "-b:a", "24k"], "ogg"),
    # VoLTE fallback / 2G. Requires an ffmpeg build with libopencore-amrnb.
    "amrnb": (["-ar", "8000", "-ac", "1", "-c:a", "libopencore_amrnb",
               "-b:a", "12.2k"], "amr"),
}


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def have_encoder(name: str) -> bool:
    """Check an encoder is actually compiled in before relying on it."""
    args, _ = CODECS[name]
    try:
        enc = args[args.index("-c:a") + 1]
    except (ValueError, IndexError):
        return False
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=20,
        ).stdout
    except Exception:  # noqa: BLE001
        return False
    return enc in out


def apply_codec(x: np.ndarray, codec: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Round-trip float32 audio through a codec and back to `sr`.

    Returns the input unchanged if ffmpeg or the encoder is unavailable, so a
    missing AMR build degrades the augmentation instead of killing the job.
    """
    if codec == "none":
        return x
    if codec not in CODECS or not have_ffmpeg():
        return x
    args, container = CODECS[codec]

    tmpdir = tempfile.mkdtemp(prefix="codec_")
    try:
        src = os.path.join(tmpdir, "in.wav")
        mid = os.path.join(tmpdir, f"mid.{container}")
        dst = os.path.join(tmpdir, "out.wav")
        sf.write(src, x, sr, subtype="PCM_16")

        enc = ["ffmpeg", "-v", "error", "-y", "-i", src, *args, mid]
        dec = ["ffmpeg", "-v", "error", "-y", "-i", mid,
               "-ar", str(sr), "-ac", "1", "-c:a", "pcm_s16le", dst]
        subprocess.run(enc, check=True, capture_output=True, timeout=120)
        subprocess.run(dec, check=True, capture_output=True, timeout=120)

        y, _ = sf.read(dst, dtype="float32")
        return y if y.ndim == 1 else y.mean(axis=1)
    except Exception as exc:  # noqa: BLE001
        print(f"  ! {codec} failed ({exc}); passing audio through", file=sys.stderr)
        return x
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def burst_packet_loss(
    x: np.ndarray,
    sr: int = SAMPLE_RATE,
    loss_rate: float = 0.03,
    burst_ms: int = 40,
    rng: Optional[random.Random] = None,
) -> np.ndarray:
    """Zero out whole 20-40 ms spans, not individual samples.

    Real loss is bursty: one lost RTP packet is one 20 ms frame, and losses
    cluster. Dropping isolated samples instead produces broadband clicks that a
    CM can learn as a shortcut -- it would score "packet loss" as "synthetic",
    which is a false-positive generator on any congested network.
    """
    rng = rng or random.Random()
    y = x.copy()
    burst = max(1, int(sr * burst_ms / 1000))
    n_bursts = int(len(y) * loss_rate / burst)
    for _ in range(n_bursts):
        if len(y) <= burst:
            break
        start = rng.randrange(0, len(y) - burst)
        y[start:start + burst] = 0.0
    return y


def augment(
    x: np.ndarray,
    sr: int = SAMPLE_RATE,
    codecs: Optional[List[str]] = None,
    loss_rate: float = 0.03,
    rng: Optional[random.Random] = None,
) -> np.ndarray:
    """Pick one codec at random, then apply burst loss. Loss first, then codec.

    Order matters: on a real call the packet is lost *after* encoding, so the
    decoder's concealment smears the gap. Encoding a pre-holed signal instead
    produces artefacts that do not occur in the wild.
    """
    rng = rng or random.Random()
    codec = rng.choice(codecs or ["none", "opus", "g711"])
    y = apply_codec(x, codec, sr)
    if loss_rate > 0:
        y = burst_packet_loss(y, sr, loss_rate, rng=rng)
    return y


# ---------------------------------------------------------------------------
def _main() -> int:
    ap = argparse.ArgumentParser(description="Codec-augment a directory of WAVs.")
    ap.add_argument("src", help="input directory")
    ap.add_argument("dst", help="output directory")
    ap.add_argument("--codecs", default="opus,g711",
                    help="comma-separated: none,opus,g711,amrnb")
    ap.add_argument("--loss", type=float, default=0.03)
    ap.add_argument("--per-file", type=int, default=1,
                    help="augmented copies per input file")
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()

    if not have_ffmpeg():
        print("ffmpeg not found on PATH. Install it -- without codec "
              "augmentation the model will not survive a real call.",
              file=sys.stderr)
        return 2

    codecs = [c.strip() for c in a.codecs.split(",") if c.strip()]
    for c in codecs:
        if c != "none" and c in CODECS and not have_encoder(c):
            print(f"  ! encoder for {c} not compiled into this ffmpeg; skipping")
            codecs = [x for x in codecs if x != c]

    os.makedirs(a.dst, exist_ok=True)
    rng = random.Random(a.seed)
    wavs = sorted(
        f for f in os.listdir(a.src) if f.lower().endswith((".wav", ".flac"))
    )
    if not wavs:
        print(f"no audio in {a.src}", file=sys.stderr)
        return 1

    for i, name in enumerate(wavs, 1):
        x, sr = sf.read(os.path.join(a.src, name), dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        for k in range(a.per_file):
            y = augment(x, sr, codecs, a.loss, rng)
            stem = os.path.splitext(name)[0]
            out = os.path.join(a.dst, f"{stem}__aug{k}.wav")
            sf.write(out, y, sr, subtype="PCM_16")
        if i % 50 == 0 or i == len(wavs):
            print(f"  {i}/{len(wavs)}")

    print(f"wrote {len(wavs) * a.per_file} files to {a.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
