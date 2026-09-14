"""
Calibrate the ASV thresholds against your own voices, on your own call path.

WHY YOU CANNOT SKIP THIS
------------------------
`ASV_TAU_ACCEPT = 0.60` and `ASV_TAU_REJECT = 0.35` in core/config.py are
placeholders. They are the numbers you find in ECAPA tutorials, measured on clean
studio VoxCeleb audio, and they do not transfer to a 16 kHz Opus leg recorded on
a phone in a noisy hall. Ship them uncalibrated and you get one of two failures:
an accept threshold so high that the enrolled speaker is flagged as a stranger,
or one so low that everyone passes.

This script measures the two distributions you actually care about -- the
enrolled speaker on held-out recordings, and other people -- and reports the
thresholds those distributions imply.

WHAT TO RECORD FIRST
--------------------
  --same   6-10 clips of the enrolled speaker that were NOT used for enrollment.
           Different sessions, different rooms, some over the actual call path.
  --other  6-10 clips of other people. Teammates are fine. Same recording chain.

Both sets should go through the codec leg before scoring, or you are calibrating
for audio you will never see:
  python ml/codec_augment.py raw/ raw_opus/ --codecs opus --loss 0.02
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
import soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from app.core import config as C  # noqa: E402
from app.engine.asv_engine import AsvEngine  # noqa: E402

WINDOW_SAMPLES = C.WINDOW_SAMPLES


def load_mono_16k(path: str) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != C.SAMPLE_RATE:
        n = int(round(len(x) * C.SAMPLE_RATE / sr))
        x = np.interp(
            np.linspace(0, len(x) - 1, max(n, 2)),
            np.arange(len(x)),
            x.astype(np.float64),
        ).astype(np.float32)
    return x


def score_dir(engine: AsvEngine, d: str) -> Tuple[List[float], int]:
    """One similarity per 2 s window, matching what runs on a live call.

    Scoring whole files instead would flatter the system: longer audio gives the
    embedder more to work with than it ever gets in production.
    """
    sims: List[float] = []
    n_files = 0
    names = sorted(
        f for f in os.listdir(d) if f.lower().endswith((".wav", ".flac", ".ogg"))
    )
    for name in names:
        try:
            x = load_mono_16k(os.path.join(d, name))
        except Exception as exc:  # noqa: BLE001
            print(f"  ! skip {name}: {exc}")
            continue
        n_files += 1
        got = 0
        for s in range(0, max(len(x) - WINDOW_SAMPLES + 1, 1), C.HOP_SAMPLES * 2):
            w = x[s:s + WINDOW_SAMPLES]
            if len(w) < WINDOW_SAMPLES:
                break
            v = engine.similarity(w)
            if v is not None:
                sims.append(float(v))
                got += 1
        print(f"  {name}: {got} windows")
    return sims, n_files


def pct(a: np.ndarray, q: float) -> float:
    return float(np.percentile(a, q)) if len(a) else float("nan")


def eer_threshold(same: np.ndarray, other: np.ndarray) -> Tuple[float, float]:
    """Sweep candidate thresholds; return (EER, threshold).

    Accepting above the threshold: FRR = fraction of same-speaker below it,
    FAR = fraction of impostors above it. The EER is where those two cross.
    """
    if not len(same) or not len(other):
        return float("nan"), float("nan")
    best = None
    for t in np.unique(np.concatenate([same, other])):
        frr = float((same < t).mean())
        far = float((other >= t).mean())
        cand = (abs(frr - far), (frr + far) / 2.0, float(t))
        if best is None or cand[0] < best[0]:
            best = cand
    _, eer, thr = best
    return eer, thr


def main() -> int:
    ap = argparse.ArgumentParser(description="Calibrate ASV thresholds.")
    ap.add_argument("--anchor", default=C.ASV_ANCHOR,
                    help="anchor .npy written by enroll.py")
    ap.add_argument("--same", required=True,
                    help="held-out clips of the ENROLLED speaker")
    ap.add_argument("--other", required=True,
                    help="clips of OTHER speakers")
    ap.add_argument("--target-far", type=float, default=0.01,
                    help="impostor accept rate to allow when picking tau_accept")
    ap.add_argument("--target-frr", type=float, default=0.01,
                    help="genuine reject rate to allow when picking tau_reject")
    a = ap.parse_args()

    engine = AsvEngine(a.anchor)
    if not engine.available:
        print("ECAPA-TDNN is not available -- install speechbrain + torch.\n"
              "Without it every similarity comes back None and the only symptom "
              "you would see below is 'not enough windows', which points at your "
              "recordings instead of your environment.", file=sys.stderr)
        return 1
    if not engine.enrolled:
        print(f"could not load anchor from {a.anchor}. Run enroll.py first.",
              file=sys.stderr)
        return 1

    print(f"scoring same-speaker clips in {a.same}")
    same_l, n_same = score_dir(engine, a.same)
    print(f"scoring other-speaker clips in {a.other}")
    other_l, n_other = score_dir(engine, a.other)

    same, other = np.array(same_l), np.array(other_l)
    if len(same) < 20 or len(other) < 20:
        print("\nNot enough windows to calibrate on. Record more audio -- "
              "20+ windows per class is the bare minimum, and thresholds from "
              "less than that are noise.", file=sys.stderr)
        if not len(same) or not len(other):
            return 1

    print("\n" + "=" * 66)
    print(f"same speaker   n={len(same):4d} windows / {n_same} files   "
          f"mean {same.mean():.3f}  p5 {pct(same, 5):.3f}  p50 {pct(same, 50):.3f}")
    print(f"other speakers n={len(other):4d} windows / {n_other} files   "
          f"mean {other.mean():.3f}  p50 {pct(other, 50):.3f}  "
          f"p95 {pct(other, 95):.3f}")

    eer, thr = eer_threshold(same, other)
    print(f"\nEER {eer * 100:.1f}% at cosine {thr:.3f}")
    if eer > 0.20:
        print("  ! The two distributions barely separate. Before touching "
              "thresholds, check that the anchor was enrolled from the same "
              "recording chain -- a mic mismatch dominates speaker identity.")

    # tau_accept: high enough that impostors rarely clear it.
    # tau_reject: low enough that the enrolled speaker rarely falls below it.
    # The gap between them is the deliberate "inconclusive" band -- the system
    # says "I do not know" instead of guessing, which is the honest behaviour and
    # keeps a bad enrollment from producing confident nonsense.
    tau_accept = pct(other, 100 * (1 - a.target_far))
    tau_reject = pct(same, 100 * a.target_frr)
    if not (tau_reject < tau_accept):
        mid = (float(np.median(same)) + float(np.median(other))) / 2.0
        tau_reject, tau_accept = mid - 0.05, mid + 0.05
        print("\n  ! Distributions overlap too much for the requested error "
              "rates; falling back to a band around the midpoint. Treat ASV as "
              "weak evidence until you have better enrollment audio.")

    print("\nPut these in backend/app/core/config.py:")
    print(f"  ASV_TAU_ACCEPT = {max(tau_accept, 0.0):.2f}")
    print(f"  ASV_TAU_REJECT = {max(tau_reject, 0.0):.2f}")
    print(f"\n(currently {C.ASV_TAU_ACCEPT:.2f} / {C.ASV_TAU_REJECT:.2f})")
    print("\nAt those values, on THIS audio:")
    print(f"  genuine below tau_reject (flagged as a stranger): "
          f"{(same < tau_reject).mean() * 100:.1f}%")
    print(f"  impostor above tau_accept (accepted as the speaker): "
          f"{(other >= tau_accept).mean() * 100:.1f}%")
    print(f"  inconclusive band width: {tau_accept - tau_reject:.3f}")
    print("\nThese are in-sample numbers on a handful of voices. They describe "
          "your setup, not the world; do not quote them as system accuracy.")
    print("Remember ASV cannot detect a clone -- a good clone is built to "
          "maximise exactly this similarity. It only tells you whether the "
          "voice matches the person you enrolled.")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
