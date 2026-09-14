#!/usr/bin/env python3
"""
Build the enrolled voiceprint (the "protected executive" anchor).

Usage:
    python tools/enroll.py --label cfo_priya samples/priya_*.wav

Guidance that matters more than the code:

* Record on the SAME phone and through the SAME call path you will demo on.
  A voiceprint captured on a laptop condenser mic and compared against 16kHz
  Opus-decoded phone audio will produce a depressed cosine similarity, and you
  will "fix" it by lowering the threshold until the system stops working.

* Use 3-5 separate takes of 8-15s each, different sentences, and average the
  embeddings. One long take encodes that day's room, mic distance, and mood.

* Get written consent from the teammate whose voice you enroll and clone, and
  put a line about it on your SIH slide. Never enroll or clone a real official,
  celebrity, or anyone outside the team. Keep generated clones out of any public
  repo -- add models/ and *_clone*.wav to .gitignore.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core import config as C  # noqa: E402
from app.engine.asv_engine import AsvEngine, cosine  # noqa: E402


def load_wav(path: str) -> np.ndarray:
    import soundfile as sf
    from scipy.signal import resample_poly

    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != C.SAMPLE_RATE:
        g = np.gcd(int(sr), C.SAMPLE_RATE)
        audio = resample_poly(audio, C.SAMPLE_RATE // g, sr // g).astype(np.float32)
    return audio.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="+", help="clean consented speech, 8-15s each")
    ap.add_argument("--label", required=True, help="e.g. cfo_priya")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    asv = AsvEngine(anchor_path="/nonexistent")   # model only, no anchor
    if not asv.available:
        print("ERROR: ECAPA-TDNN unavailable. Install speechbrain + torch.")
        return 1

    embeddings = []
    for path in args.wavs:
        audio = load_wav(path)
        dur = audio.size / C.SAMPLE_RATE
        if dur < 5.0:
            print(f"  SKIP {os.path.basename(path)}: only {dur:.1f}s, need >= 5s")
            continue
        emb = asv.embed(audio)
        if emb is None:
            print(f"  FAIL {os.path.basename(path)}")
            continue
        embeddings.append(emb)
        print(f"  ok   {os.path.basename(path)}  {dur:.1f}s")

    if len(embeddings) < 2:
        print("ERROR: need at least 2 usable takes.")
        return 1

    # Consistency check across takes. Low pairwise similarity means the takes are
    # not comparable (different mic, clipping, background music) and the anchor
    # will be a blurred average that matches nothing well.
    pair_sims = [
        cosine(embeddings[i], embeddings[j])
        for i in range(len(embeddings))
        for j in range(i + 1, len(embeddings))
    ]
    mean_pair = float(np.mean(pair_sims))
    print(f"\nwithin-speaker consistency: mean pairwise cosine {mean_pair:.3f}")
    if mean_pair < 0.55:
        print("  WARNING: takes disagree. Re-record on one device, one setting.")

    anchor = np.mean(embeddings, axis=0)
    anchor = anchor / (np.linalg.norm(anchor) + 1e-9)

    out = args.out or os.path.join(C.MODELS_DIR, f"anchor_{args.label}.npy")
    # dirname("anchor.npy") is "", and makedirs("") raises FileNotFoundError.
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.save(
        out,
        {"embedding": anchor, "label": args.label, "n_takes": len(embeddings),
         "mean_pairwise_cosine": mean_pair},
        allow_pickle=True,
    )
    print(f"\nwrote {out}  (dim={anchor.size})")
    print(f"point the backend at it:  cp {out} {C.ASV_ANCHOR}")
    print("\nNEXT: run tools/calibrate_asv.py to set ASV_TAU_* from your own data.")
    print("Do NOT ship the placeholder thresholds in config.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
