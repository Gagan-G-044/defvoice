"""
Train the CM head on cached SSL features.

Scope: this trains a ~1M-parameter classifier head on frozen Wav2Vec2-XLS-R
features. It does not fine-tune the SSL front-end. On a 4 GB RTX 3050 that is
not a compromise you should feel bad about -- full fine-tuning needs gradient
checkpointing and hours per epoch, and with a week of runway it is how teams end
up with no working demo.

TWO RULES THAT DECIDE WHETHER THE NUMBER MEANS ANYTHING
------------------------------------------------------
1. Split by source file, never by window. Four crops of one utterance in both
   train and validation gives you a validation EER of ~2% and a system that
   fails on the first unseen speaker. This script groups by `src`.

2. Keep the TTS systems you demo with out of training entirely. If you generate
   the demo clips with XTTSv2 or F5-TTS, those systems belong only in the test
   split. Otherwise the headline result is "the model recognises the vocoder it
   was trained on", which is true of every published CM and is exactly the
   criticism a technical judge will make.

Report the equal-error rate, not accuracy. Accuracy on an imbalanced spoof corpus
is a number that flatters a model which has learned to say "spoof" every time.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "backend"))

from app.engine.cm_head import LinearHead  # noqa: E402

# Class-index convention. The backend defaults to spoof at index 0 via
# DEFVOICE_CM_SPOOF_INDEX; keep both sides in agreement or the gauge reads green
# on every clone. This is written into the checkpoint so it cannot drift.
SPOOF_INDEX = 0
BONAFIDE_INDEX = 1


def load_cache(cache_dir: str) -> Tuple[dict, List[dict]]:
    with open(os.path.join(cache_dir, "index.json"), "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    return meta, meta["items"]


def group_split(
    items: List[dict], val_frac: float, seed: int
) -> Tuple[List[dict], List[dict]]:
    """Split on source file so no utterance appears on both sides."""
    by_src: Dict[str, List[dict]] = defaultdict(list)
    for it in items:
        by_src[it.get("src", it["file"])].append(it)
    keys = sorted(by_src)
    random.Random(seed).shuffle(keys)
    n_val = max(1, int(len(keys) * val_frac))
    val_keys = set(keys[:n_val])
    train, val = [], []
    for k in keys:
        (val if k in val_keys else train).extend(by_src[k])
    return train, val


def stack(cache_dir: str, items: List[dict]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load every cached window into RAM.

    A modest ASVspoof subset at 4 crops/file is a few GB in fp16, which fits.
    If you scale past RAM, swap this for a Dataset that mmaps each .npy -- the
    training loop below does not care.
    """
    xs, ys = [], []
    for it in items:
        arr = np.load(os.path.join(cache_dir, it["file"]))
        xs.append(torch.from_numpy(arr.astype(np.float32)))
        ys.extend([it["label"]] * arr.shape[0])
    return torch.cat(xs, dim=0), torch.tensor(ys, dtype=torch.long)


def eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """Equal error rate and its threshold. `scores` = P(spoof), `labels` 1=spoof.

    Computed by sweeping every distinct score as a threshold: exact, no
    interpolation, and fast enough for validation-set sizes.
    """
    order = np.argsort(-scores)
    s, y = scores[order], labels[order]
    n_pos = max(int((y == 1).sum()), 1)
    n_neg = max(int((y == 0).sum()), 1)
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    fnr = 1.0 - tp / n_pos      # missed spoofs
    fpr = fp / n_neg            # bonafide flagged as spoof
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fnr[i] + fpr[i]) / 2.0), float(s[i])


def main() -> int:
    ap = argparse.ArgumentParser(description="Train the CM head.")
    ap.add_argument("cache", help="directory produced by cache_features.py")
    ap.add_argument("--out", default="cm_head.pt")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)

    device = "cpu" if a.cpu or not torch.cuda.is_available() else "cuda"
    meta, items = load_cache(a.cache)
    hidden = int(meta["hidden"])
    print(f"device={device} hidden={hidden} layer={meta['layer']} "
          f"ssl={meta['model']}")

    tr_items, va_items = group_split(items, a.val_frac, a.seed)
    Xtr, ytr = stack(a.cache, tr_items)
    Xva, yva = stack(a.cache, va_items)
    print(f"train {len(ytr)} windows / {len(tr_items)} files · "
          f"val {len(yva)} windows / {len(va_items)} files")

    # Two logits, spoof at SPOOF_INDEX. Class weights counter the spoof-heavy
    # ratio in ASVspoof, which otherwise trains a model that just says spoof.
    head = LinearHead(in_dim=hidden, n_classes=2).to(device)
    n_spoof = int((ytr == 1).sum())
    n_bona = int((ytr == 0).sum())
    w = torch.ones(2, device=device)
    if n_spoof and n_bona:
        total = n_spoof + n_bona
        w[SPOOF_INDEX] = total / (2.0 * n_spoof)
        w[BONAFIDE_INDEX] = total / (2.0 * n_bona)
    print(f"class weights: spoof {w[SPOOF_INDEX]:.2f} bonafide {w[BONAFIDE_INDEX]:.2f}")

    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    def to_target(y: torch.Tensor) -> torch.Tensor:
        """Dataset label (1=spoof) -> class index at SPOOF_INDEX."""
        return torch.where(
            y == 1,
            torch.full_like(y, SPOOF_INDEX),
            torch.full_like(y, BONAFIDE_INDEX),
        )

    best = (1.0, -1)   # (eer, epoch)
    for ep in range(1, a.epochs + 1):
        head.train()
        perm = torch.randperm(Xtr.shape[0])
        running = 0.0
        for s in range(0, len(perm), a.batch):
            idx = perm[s:s + a.batch]
            xb = Xtr[idx].to(device)
            yb = to_target(ytr[idx].to(device))
            opt.zero_grad(set_to_none=True)
            loss = crit(head(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            opt.step()
            running += float(loss) * len(idx)
        sched.step()

        head.eval()
        scores = []
        with torch.no_grad():
            for s in range(0, Xva.shape[0], 256):
                logits = head(Xva[s:s + 256].to(device))
                p = torch.softmax(logits.float(), dim=-1)[:, SPOOF_INDEX]
                scores.append(p.cpu().numpy())
        sc = np.concatenate(scores) if scores else np.zeros(0)
        e, thr = eer(sc, yva.numpy())
        print(f"epoch {ep:2d}  loss {running / max(len(perm), 1):.4f}  "
              f"val EER {e * 100:.2f}%  thr {thr:.3f}")

        if e < best[0]:
            best = (e, ep)
            torch.save(
                {
                    # Key names are dictated by cm_engine.TorchCm.__init__.
                    # Do not "tidy" them -- the loader reads exactly these.
                    "head_state": head.state_dict(),
                    "encoder_name": meta["model"],
                    "layer": int(meta["layer"]),
                    # Metadata: not read by the backend, but this is the only
                    # place the provenance of a checkpoint survives.
                    "in_dim": hidden,
                    "n_classes": 2,
                    "spoof_index": SPOOF_INDEX,
                    "sample_rate": meta["sample_rate"],
                    "window_sec": meta["window_sec"],
                    "val_eer": e,
                    "val_threshold": thr,
                    "n_train_files": len(tr_items),
                    "n_val_files": len(va_items),
                },
                a.out,
            )

    print(f"\nbest val EER {best[0] * 100:.2f}% at epoch {best[1]} -> {a.out}")
    print("\nTo use it, drop the checkpoint where the engine looks for it:")
    print(f"  copy {os.path.abspath(a.out)} -> backend/models/cm_head.pt")
    print(f"  set DEFVOICE_CM_SPOOF_INDEX={SPOOF_INDEX}")
    print("  set DEFVOICE_ALLOW_STUB=0")
    print("The encoder name and layer travel inside the checkpoint, so the "
          "inference front-end cannot drift from the training front-end.")
    print("\nThen confirm the polarity on held-out audio before trusting it:")
    print("  python backend/tools/check_cm_polarity.py --genuine <dir> "
          "--spoof <dir>")
    print("\nA validation EER from one corpus is not a claim about the world. "
          "Quote it with the corpus name and the systems it was measured on, "
          "and expect it to roughly double on unseen TTS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
