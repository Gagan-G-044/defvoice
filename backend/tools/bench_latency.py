"""
Measure whether this laptop can actually keep up.

The demo promises a verdict every HOP_SEC seconds. That only holds if scoring one
2 s window finishes in under HOP_SEC. When it does not, the analyzer's
latest-wins policy silently drops windows: the app still looks live, the gauge
still moves, and the real detection latency is quietly two or three times what
you are claiming on stage. This script tells you which regime you are in before a
judge asks.

Run it on the machine you will demo on, with the models you will demo with, on
mains power. A 3050 in battery-saver mode clocks down hard.

  python backend/tools/bench_latency.py
  python backend/tools/bench_latency.py --wav sample.wav --n 60
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from typing import Dict, List

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from app.core import config as C  # noqa: E402
from app.engine.pipeline import EngineBundle  # noqa: E402


def synthetic_window(rng: np.random.Generator) -> np.ndarray:
    """Speech-ish noise: a few harmonics plus noise, at speech-like level.

    Not real speech, but the compute cost of a transformer forward pass does not
    depend on content, and this keeps the benchmark runnable with no data.
    """
    t = np.arange(C.WINDOW_SAMPLES, dtype=np.float32) / C.SAMPLE_RATE
    f0 = 120.0 + rng.uniform(-20, 20)
    x = sum(
        (0.5 / k) * np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 6.28))
        for k in range(1, 9)
    )
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t)
    x = (x * env + 0.02 * rng.standard_normal(len(t))).astype(np.float32)
    return (x / (np.abs(x).max() + 1e-9) * 0.3).astype(np.float32)


def windows_from_wav(path: str, limit: int) -> List[np.ndarray]:
    import soundfile as sf

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
    out = []
    for s in range(0, len(x) - C.WINDOW_SAMPLES + 1, C.HOP_SAMPLES):
        out.append(x[s:s + C.WINDOW_SAMPLES])
        if len(out) >= limit:
            break
    return out


def summarize(name: str, samples: List[float], budget_ms: float) -> Dict:
    if not samples:
        return {"name": name, "n": 0}
    s = sorted(samples)
    p50 = statistics.median(s)
    p95 = s[min(int(0.95 * len(s)), len(s) - 1)]
    return {
        "name": name,
        "n": len(s),
        "p50": p50,
        "p95": p95,
        "max": s[-1],
        "over_budget": sum(1 for v in s if v > budget_ms),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark per-window inference.")
    ap.add_argument("--wav", default="", help="score real audio instead of noise")
    ap.add_argument("--n", type=int, default=40, help="windows to score")
    ap.add_argument("--warmup", type=int, default=3,
                    help="discarded; the first CUDA call pays kernel compilation")
    a = ap.parse_args()

    budget_ms = C.HOP_SEC * 1000.0
    print(f"window {C.WINDOW_SEC}s  hop {C.HOP_SEC}s  "
          f"budget {budget_ms:.0f} ms per window")

    t0 = time.perf_counter()
    eng = EngineBundle()
    print(f"engines loaded in {time.perf_counter() - t0:.1f}s: {eng.status()}")
    if eng.degraded:
        print("\n  ! Stub CM. These timings are meaningless as a capacity "
              "estimate -- the stub is a spectral heuristic and costs almost "
              "nothing. Re-run with real weights loaded.")

    rng = np.random.default_rng(7)
    if a.wav:
        wins = windows_from_wav(a.wav, a.n)
        if not wins:
            print(f"{a.wav} is shorter than one {C.WINDOW_SEC}s window")
            return 1
    else:
        wins = [synthetic_window(rng) for _ in range(a.n)]
    print(f"scoring {len(wins)} windows"
          f"{' from ' + os.path.basename(a.wav) if a.wav else ' of synthetic audio'}")

    for w in wins[: a.warmup]:
        eng.vad.is_voiced(w)
        eng.cm.score(w)
        eng.asv.similarity(w)

    vad_ms: List[float] = []
    cm_ms: List[float] = []
    asv_ms: List[float] = []
    total_ms: List[float] = []
    voiced_n = 0

    for w in wins:
        t_start = time.perf_counter()
        t = time.perf_counter()
        voiced, _ = eng.vad.is_voiced(w)
        vad_ms.append((time.perf_counter() - t) * 1000)

        # Mirror _score_window: an unvoiced window skips CM and ASV entirely,
        # which is where most of the headroom on a real call comes from.
        if voiced:
            voiced_n += 1
            t = time.perf_counter()
            eng.cm.score(w)
            cm_ms.append((time.perf_counter() - t) * 1000)
            t = time.perf_counter()
            eng.asv.similarity(w)
            asv_ms.append((time.perf_counter() - t) * 1000)
        total_ms.append((time.perf_counter() - t_start) * 1000)

    rows = [
        summarize("VAD", vad_ms, budget_ms),
        summarize("CM", cm_ms, budget_ms),
        summarize("ASV", asv_ms, budget_ms),
        summarize("TOTAL per window", total_ms, budget_ms),
    ]
    print("\n" + "=" * 66)
    print(f"{'stage':<18}{'n':>5}{'p50 ms':>10}{'p95 ms':>10}"
          f"{'max ms':>10}{'over':>7}")
    for r in rows:
        if not r["n"]:
            print(f"{r['name']:<18}{0:>5}   (never ran)")
            continue
        print(f"{r['name']:<18}{r['n']:>5}{r['p50']:>10.1f}{r['p95']:>10.1f}"
              f"{r['max']:>10.1f}{r['over_budget']:>7}")
    print(f"\nvoiced windows: {voiced_n}/{len(wins)}")

    tot = rows[-1]
    if tot["n"]:
        rtf = tot["p95"] / budget_ms
        print(f"p95 uses {rtf * 100:.0f}% of the {budget_ms:.0f} ms budget")
        if rtf < 0.6:
            print("\nVERDICT: comfortable. Every window gets scored; the "
                  "detection latency you measure in the tests is the latency "
                  "you will see.")
        elif rtf < 1.0:
            print("\nVERDICT: tight. It fits, but a background Chrome window or "
                  "a thermally throttled GPU will push it over. Close "
                  "everything else before demoing, and keep the laptop plugged "
                  "in.")
        else:
            print("\nVERDICT: over budget. Windows WILL be dropped and your "
                  "real detection latency is worse than the test suite says.")
            print("Cheapest fixes, in order:")
            print("  1. Raise HOP_SEC to 1.0 in core/config.py. Halves the load; "
                  "costs about half a second of detection latency. Re-run "
                  "tests/test_risk_engine.py and quote the new number.")
            print("  2. Export the CM to ONNX and run it on CUDAExecutionProvider "
                  "-- usually 2-3x faster than the PyTorch path.")
            print("  3. Use a smaller SSL front-end (wav2vec2-base over "
                  "XLS-R-300m). Costs accuracy, especially cross-lingual.")
            print("  Do not 'fix' it by removing the ASV branch; that is the "
                  "cheap stage.")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
