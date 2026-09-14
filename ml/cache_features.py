"""
Cache frozen SSL features once, train the head many times.

This is the single biggest reason the CM is trainable on a 4 GB RTX 3050. A
Wav2Vec2-XLS-R forward pass dominates step time and its weights never change, so
running it every epoch is pure waste. Extract once to disk, then each head epoch
is a few seconds of matrix multiplies and you can afford to actually tune things.

MUST MATCH INFERENCE
--------------------
The model id, the hidden layer, and the normalisation here have to be identical
to what `backend/app/engine/cm_engine.py:TorchCm` does at inference time. A
mismatch produces a head that trains beautifully and scores noise on a live call,
and the symptom -- good validation EER, useless demo -- looks like a data problem
rather than a preprocessing one.

The model id and layer are handled for you: `train_cm_head.py` copies them out of
`index.json` and into the checkpoint as `encoder_name` and `layer`, and `TorchCm`
reads them back from there. So whatever you pass to `--model` and `--layer` below
is what inference will use. The normalisation is the part with no such guardrail
-- if you change the mean/variance scheme in `extract()`, change it in `TorchCm`
too.

Layer choice: middle layers of XLS-R carry more phonetic/acoustic detail and less
semantic abstraction, which is what spoof detection needs. Layer 5 is a common
default in the ASVspoof literature. If you have time, cache 4-8 and let
WeightedLayerSum learn the mix.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List, Tuple

import numpy as np
import soundfile as sf
import torch

SAMPLE_RATE = 16_000
WINDOW_SEC = 2.0
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SEC)

SSL_MODEL = os.environ.get("DEFVOICE_SSL_MODEL", "facebook/wav2vec2-xls-r-300m")
SSL_LAYER = int(os.environ.get("DEFVOICE_SSL_LAYER", "5"))


def load_audio(path: str) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SAMPLE_RATE:
        # Linear interpolation is adequate here: everything is already low-pass
        # limited by the codec leg, and a proper polyphase resample would pull in
        # another dependency for no measurable gain.
        n = int(round(len(x) * SAMPLE_RATE / sr))
        if n <= 1:
            return np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        x = np.interp(
            np.linspace(0, len(x) - 1, n, dtype=np.float64),
            np.arange(len(x), dtype=np.float64),
            x.astype(np.float64),
        ).astype(np.float32)
    return x


def crops(x: np.ndarray, max_windows: int, hop_sec: float) -> List[np.ndarray]:
    """Strided 2 s crops. Deterministic, so re-running the cache is idempotent."""
    if len(x) < WINDOW_SAMPLES:
        pad = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        pad[: len(x)] = x
        return [pad]
    hop = max(1, int(SAMPLE_RATE * hop_sec))
    out: List[np.ndarray] = []
    for start in range(0, len(x) - WINDOW_SAMPLES + 1, hop):
        out.append(x[start:start + WINDOW_SAMPLES])
        if len(out) >= max_windows:
            break
    return out


def read_manifest(path: str) -> List[Tuple[str, int]]:
    """`path<TAB or ,>label` per line; label 0 = bonafide, 1 = spoof.

    Comments and a header row starting with 'path' are skipped.
    """
    items: List[Tuple[str, int]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or line.lower().startswith("path"):
                continue
            parts = line.split("\t") if "\t" in line else line.split(",")
            if len(parts) < 2:
                continue
            items.append((parts[0].strip(), int(parts[1].strip())))
    return items


@torch.no_grad()
def extract(model, batch: np.ndarray, device: str, layer: int) -> np.ndarray:
    """(B, 32000) float32 -> (B, T, H) float16."""
    wav = torch.from_numpy(batch).to(device)
    # Per-utterance zero-mean/unit-variance. Same normalisation as inference.
    wav = (wav - wav.mean(dim=-1, keepdim=True)) / (
        wav.std(dim=-1, keepdim=True) + 1e-7
    )
    out = model(wav, output_hidden_states=True)
    # hidden_states[0] is the conv feature projection; transformer block i is
    # hidden_states[i]. Layer indices in papers usually mean the latter.
    h = out.hidden_states[layer]
    return h.detach().float().cpu().numpy().astype(np.float16)


def main() -> int:
    ap = argparse.ArgumentParser(description="Cache frozen SSL features.")
    ap.add_argument("manifest", help="TSV/CSV of path,label (0=bonafide,1=spoof)")
    ap.add_argument("out", help="output directory")
    ap.add_argument("--layer", type=int, default=SSL_LAYER)
    ap.add_argument("--model", default=SSL_MODEL)
    ap.add_argument("--max-windows", type=int, default=4,
                    help="crops per file; 4 keeps class balance without bloat")
    ap.add_argument("--hop-sec", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=4,
                    help="4 x 2s fits 4 GB VRAM with XLS-R 300m in fp16")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()

    from transformers import Wav2Vec2Model  # imported late; heavy

    device = "cpu" if a.cpu or not torch.cuda.is_available() else "cuda"
    print(f"device={device} model={a.model} layer={a.layer}")

    model = Wav2Vec2Model.from_pretrained(a.model)
    model.eval().to(device)
    if device == "cuda":
        model.half()
    for p in model.parameters():
        p.requires_grad_(False)

    items = read_manifest(a.manifest)
    if not items:
        print("empty manifest")
        return 1
    n_spoof = sum(1 for _, y in items if y == 1)
    print(f"{len(items)} files ({n_spoof} spoof, {len(items) - n_spoof} bonafide)")

    os.makedirs(a.out, exist_ok=True)
    index = []
    hidden = None

    for i, (path, label) in enumerate(items, 1):
        try:
            x = load_audio(path)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! skip {path}: {exc}")
            continue
        windows = crops(x, a.max_windows, a.hop_sec)

        feats = []
        dtype = np.float16 if device == "cuda" else np.float32
        for s in range(0, len(windows), a.batch):
            chunk = np.stack(windows[s:s + a.batch]).astype(dtype)
            feats.append(extract(model, chunk, device, a.layer))
        arr = np.concatenate(feats, axis=0)

        stem = f"{i:06d}"
        np.save(os.path.join(a.out, f"{stem}.npy"), arr)
        index.append({
            "file": f"{stem}.npy",
            "label": int(label),
            "n_windows": int(arr.shape[0]),
            "src": path,
        })
        if hidden is None:
            hidden = int(arr.shape[-1])
        if i % 25 == 0 or i == len(items):
            print(f"  {i}/{len(items)}")

    meta = {
        "model": a.model,
        "layer": a.layer,
        "hidden": hidden,
        "sample_rate": SAMPLE_RATE,
        "window_sec": WINDOW_SEC,
        "items": index,
    }
    if not index:
        # Every file failed to load. Writing hidden=null here would make
        # train_cm_head.py die later on int(None) with a traceback that points at
        # the trainer instead of at the manifest, which is the actual problem.
        print("\nNo files were cached -- every path in the manifest failed to "
              "load. Check the paths are relative to your current directory, not "
              "to the manifest.")
        return 1
    with open(os.path.join(a.out, "index.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    total = sum(it["n_windows"] for it in index)
    print(f"cached {total} windows, hidden={hidden} -> {a.out}/index.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
