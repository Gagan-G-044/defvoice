#!/usr/bin/env python3
"""
Polarity + sanity check for the CM engine. RUN THIS BEFORE EVERY DEMO.

    python tools/check_cm_polarity.py --genuine real/*.wav --spoof clones/*.wav

ASVspoof reference implementations disagree about whether output index 0 is
"spoof" or "bonafide", and different training scripts flip it again. If you get
it backwards the gauge stays green through the entire attack and goes red on your
own teammate -- a failure mode that looks exactly like a working system right up
until the moment it matters.

This script reports mean P(synthetic) for each class, tells you whether the sign
is right, and prints a threshold-free separability number (AUC) so you know
whether the model is doing anything at all. Fix DEFVOICE_CM_SPOOF_INDEX (0 or 1)
until genuine scores low and spoof scores high.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core import config as C  # noqa: E402
from app.engine.cm_engine import SPOOF_CLASS_INDEX, CmEngine  # noqa: E402
from app.engine.vad_engine import VadEngine  # noqa: E402


def load(path: str) -> np.ndarray:
    import soundfile as sf
    from scipy.signal import resample_poly

    a, sr = sf.read(path, dtype="float32", always_2d=False)
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != C.SAMPLE_RATE:
        g = np.gcd(int(sr), C.SAMPLE_RATE)
        a = resample_poly(a, C.SAMPLE_RATE // g, sr // g)
    return a.astype(np.float32)


def score_files(cm: CmEngine, paths: list, vad: VadEngine | None = None) -> np.ndarray:
    """Score every voiced 2s window and keep the per-file median."""
    out = []
    for p in paths:
        if vad is not None:
            vad.reset()
        audio = load(p)
        if audio.size < C.WINDOW_SAMPLES:
            audio = np.pad(audio, (0, C.WINDOW_SAMPLES - audio.size))
        
        scores = []
        for i in range(0, audio.size - C.WINDOW_SAMPLES + 1, C.HOP_SAMPLES):
            win = audio[i : i + C.WINDOW_SAMPLES]
            if vad is not None:
                voiced, _ = vad.is_voiced(win)
                if not voiced:
                    continue
            s = cm.score(win)
            if s is not None:
                scores.append(s)

        if scores:
            out.append(float(np.median(scores)))
            print(f"  {os.path.basename(p)[:44]:<46} {out[-1]:.3f}")
        else:
            print(f"  {os.path.basename(p)[:44]:<46} FAILED (no voiced segments)")
    return np.asarray(out)


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney U / rank AUC. 0.5 == coin flip, 1.0 == perfect."""
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(np.float64) + 1
    r_pos = ranks[: pos.size].sum()
    return float((r_pos - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def expand(patterns: list) -> list:
    files: list = []
    for p in patterns:
        files.extend(sorted(glob.glob(p)))
    return files


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--genuine", nargs="+", required=True)
    ap.add_argument("--spoof", nargs="+", required=True)
    ap.add_argument("--no-vad", action="store_true", help="Disable VAD gating")
    args = ap.parse_args()

    cm = CmEngine()
    vad = None if args.no_vad else VadEngine()
    vad_status = "disabled" if vad is None else ("silero" if vad.available else "energy-fallback")
    print(f"backend={cm.name}  degraded={cm.degraded}  vad={vad_status}  "
          f"SPOOF_CLASS_INDEX={SPOOF_CLASS_INDEX}\n")
    if cm.degraded:
        print("!! stub CM -- these numbers describe a spectral heuristic, "
              "not a detector.\n")

    print("genuine:")
    g = score_files(cm, expand(args.genuine), vad=vad)
    print("\nspoof:")
    s = score_files(cm, expand(args.spoof), vad=vad)

    if g.size == 0 or s.size == 0:
        print("\nERROR: need at least one file of each class.")
        return 1

    a = auc(s, g)
    print("\n" + "=" * 62)
    print(f"mean P(synthetic)  genuine={g.mean():.3f}   spoof={s.mean():.3f}")
    print(f"AUC (spoof vs genuine) = {a:.3f}")

    if s.mean() > g.mean() + 0.15:
        print("POLARITY OK  -- spoof scores higher, as it must.")
    elif g.mean() > s.mean() + 0.15:
        print(f"POLARITY INVERTED -- set DEFVOICE_CM_SPOOF_INDEX="
              f"{1 - SPOOF_CLASS_INDEX} and re-run.")
    else:
        print("NOT SEPARATING -- the classes overlap. This is not a polarity "
              "problem, the model is not discriminating. Do not demo.")

    if a < 0.75:
        print(f"AUC {a:.3f} is too low to demo. Check that your eval audio went "
              "through the same codec chain you trained with.")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
