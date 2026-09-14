# ml/ — training the countermeasure

Four scripts, run in this order. Nothing here is imported by the backend; the
only artefact that crosses over is `cm_head.pt`.

```
raw audio ──► codec_augment.py ──► cache_features.py ──► train_cm_head.py ──► cm_head.pt
                                                                                  │
generate_eval_clones.py ──► (unseen attack split, never trained on) ──────────────┘
                                                                          evaluated with
                                                              backend/tools/check_cm_polarity.py
```

## The two rules that decide whether your number means anything

**Split by source file, never by window.** A 2-second window from the same
recording as its neighbour is not an independent sample. Split at the window
level and the model memorises recordings, validation EER collapses to something
beautiful, and the demo fails. `train_cm_head.py` groups by the `src` field that
`cache_features.py` writes into `index.json`, so this is handled — but if you
build your own manifests, keep one speaker/recording on one side of the split.

**Nothing you demo may appear in training.** The clips from
`generate_eval_clones.py` are your *unseen* attack. A CM that has seen its
vocoder during training reports an EER that describes nothing. `guard_output_dir`
in that script refuses to write into any path containing `train`, which is
enforcement rather than a comment, but it cannot stop you from copying files
around afterwards.

## 1. Codec augmentation

Do this first, on everything. A CM trained on clean audio collapses on a real
call, because the artefacts it learned live in the 4–8 kHz band that G.711 and
AMR-NB throw away.

```bash
python ml/codec_augment.py data/raw/ data/aug/ --codecs opus,g711 --loss 0.03
```

Requires `ffmpeg` on PATH. The script checks which encoders your build actually
has and drops the ones it doesn't (`libopencore_amrnb` is commonly absent) rather
than silently producing unprocessed copies. Packet loss is applied *after*
encoding, in whole 40 ms bursts, because that is what a jitter buffer sees —
zeroing scattered individual samples instead produces broadband clicks that a CM
will happily learn as its shortcut.

## 2. Cache the frozen features

This is the step that makes the whole thing trainable on a 4 GB RTX 3050. The
XLS-R forward pass dominates step time and its weights never change, so paying
for it once per epoch is pure waste.

```bash
python ml/cache_features.py manifest.tsv data/feat/ --layer 5 --batch 4
```

`manifest.tsv` is `path<TAB>label` per line, **label 0 = bonafide, 1 = spoof**.
Middle layers of XLS-R carry the phonetic and acoustic detail spoof detection
needs; layer 5 is the usual default in the ASVspoof literature. The model id and
layer are recorded in `index.json` and travel into the checkpoint, so inference
uses whatever you chose here.

The one thing with no guardrail is the waveform normalisation in `extract()`. It
must stay identical to `TorchCm.score()` in
`backend/app/engine/cm_engine.py` (per-utterance zero-mean, unit-variance). A
mismatch there gives you a healthy validation EER and a head that scores noise on
a live call, and it looks like a data problem rather than a preprocessing one.

## 3. Train the head

```bash
python ml/train_cm_head.py data/feat/ --out cm_head.pt --epochs 30
copy cm_head.pt ..\backend\models\cm_head.pt
```

Roughly 0.9 M trainable parameters over cached features: seconds per epoch, so
you can afford to actually tune it. Read the printed **EER**, not accuracy —
accuracy flatters a model that always answers "spoof", and with a class-imbalanced
manifest it will look excellent while being useless.

Then set the polarity and turn off the stub:

```bash
set DEFVOICE_CM_SPOOF_INDEX=0
set DEFVOICE_ALLOW_STUB=0
python backend/tools/check_cm_polarity.py --genuine real.wav --spoof clone.wav
```

Do not skip the polarity check. ASVspoof codebases disagree on whether class 0 is
spoof or bonafide; getting it backwards makes the gauge go green on every clone,
which is the single most common way this demo dies on stage.

## 4. Generate the unseen attack

```bash
python ml/generate_eval_clones.py --ref teammate.wav --out data/eval_unseen/ \
    --i-have-written-consent
```

Clone a teammate who has agreed in writing to this specific use. Never a public
official, never a celebrity, never a recording found online — a team that demos a
cloned politician has built the exact harm the problem statement is about. Every
output directory gets a `PROVENANCE.json` recording the engine, its version and a
hash of the reference audio, because "what was this evaluated against?" is the
first question a judge asks about a number, and three weeks later you will not
remember.

Pass these through `codec_augment.py` before scoring them, then score with
`check_cm_polarity.py`.

## What to expect

Expect the unseen-TTS EER to be roughly **double** your validation EER. That gap
is the honest result and it is the number to put on the slide, with the vocoders
you tested named next to it. A single figure with no attack list attached is not a
claim anybody should believe, including you.

None of this makes the system a universal detector. It makes it a detector that
works on the attacks you measured, on the call path you measured them over.
