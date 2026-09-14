"""
Generate the unseen-TTS evaluation clips (and the demo clips).

WHY THIS IS A SEPARATE SCRIPT WITH A CONSENT FLAG
-------------------------------------------------
Everything this produces is a synthetic copy of a real person's voice. Two rules
follow, and they are not decoration:

1. The person whose voice is cloned must have agreed, in writing, to this
   specific use. Clone a teammate. Never a public official, never a celebrity,
   never a recording you found online. A team that demos a cloned politician has
   built the exact harm the problem statement is about.

2. Nothing generated here may enter the CM training split. Ever. These systems
   are the *unseen* attack in your evaluation, and a CM that has seen its
   vocoder in training reports an EER that means nothing. This script refuses to
   write into any directory whose path contains "train".

Output is stamped with a PROVENANCE.json recording the engine, its version, the
reference audio hash and the date. Without that you cannot answer "what was this
evaluated against?" three weeks later, and that is the first question a judge
asks about a number.

The heavy TTS packages are not imported unless used, and are not dependencies of
the backend.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys
from typing import List

# Demo phrases. Short, in-domain, and the kind of thing a real vishing call says.
DEFAULT_PHRASES = [
    ("hi", "आठ लाख रुपये तुरंत इस खाते में भेजो।"),
    ("en", "Read me the OTP now, the vendor payment is stuck."),
    ("en", "I am in a meeting, just approve the transfer and I will explain later."),
    ("kn", "ತಕ್ಷಣ ಹಣ ಕಳುಹಿಸಿ, ಇದು ತುರ್ತು."),
]


def sha256(path: str, limit: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(min(1 << 20, limit))
            if not b:
                break
            h.update(b)
            limit -= len(b)
            if limit <= 0:
                break
    return h.hexdigest()[:16]


def guard_output_dir(path: str) -> None:
    """Structural enforcement of the data-leakage rule.

    A comment saying "do not put these in training" is advice. This is a check.
    """
    norm = os.path.normpath(os.path.abspath(path)).replace("\\", "/").lower()
    for bad in ("/train", "train/", "_train"):
        if bad in norm:
            raise SystemExit(
                f"Refusing to write generated clones into '{path}'.\n"
                "These clips are the unseen test split. If they reach training, "
                "your reported EER stops meaning anything.\n"
                "Use something like data/eval_unseen/ instead."
            )


def synth_xtts(ref_wav: str, phrases: List[tuple], out_dir: str) -> List[str]:
    """Coqui XTTS v2 via the `TTS` package (pip install TTS).

    Runs on CPU if no GPU is free. XTTS is ~2 GB of weights on first use.
    """
    from TTS.api import TTS  # noqa: N811 - imported late and optionally

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        device = "cpu"

    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    written = []
    for i, (lang, text) in enumerate(phrases, 1):
        # XTTS has no Kannada model; fall back to Hindi phonetics, which is what
        # a real attacker would also have to do. Note it in the filename so the
        # evaluation does not silently mislabel the language.
        eff = lang if lang in ("en", "hi") else "hi"
        suffix = "" if eff == lang else f"_as{eff}"
        out = os.path.join(out_dir, f"xtts_{i:02d}_{lang}{suffix}.wav")
        print(f"  [{i}/{len(phrases)}] {lang} -> {os.path.basename(out)}")
        tts.tts_to_file(
            text=text, speaker_wav=ref_wav, language=eff, file_path=out
        )
        written.append(out)
    return written


def synth_cmd(cmd_template: str, phrases: List[tuple], out_dir: str,
              ref_wav: str) -> List[str]:
    """Escape hatch for any other engine, e.g. F5-TTS.

    The template is expanded with {text} {lang} {ref} {out} and run as a shell
    command. Keeps this script from growing a dependency per TTS system.
    """
    written = []
    for i, (lang, text) in enumerate(phrases, 1):
        out = os.path.join(out_dir, f"cmd_{i:02d}_{lang}.wav")
        cmd = cmd_template.format(text=text, lang=lang, ref=ref_wav, out=out)
        print(f"  [{i}/{len(phrases)}] {cmd}")
        subprocess.run(cmd, shell=True, check=True)
        written.append(out)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate unseen-TTS clone clips for evaluation and demo."
    )
    ap.add_argument("--ref", required=True,
                    help="reference recording of the CONSENTING speaker (wav)")
    ap.add_argument("--out", required=True,
                    help="output directory (must not be a training dir)")
    ap.add_argument("--engine", default="xtts", choices=["xtts", "cmd"])
    ap.add_argument("--cmd", default="",
                    help="for --engine cmd: template with {text} {lang} {ref} {out}")
    ap.add_argument("--phrases", default="",
                    help="optional file, one 'lang<TAB>text' per line")
    ap.add_argument("--i-have-written-consent", action="store_true",
                    help="required; asserts the speaker agreed to this use")
    a = ap.parse_args()

    if not a.i_have_written_consent:
        print(
            "Refusing to run without --i-have-written-consent.\n\n"
            "This produces a synthetic copy of a real person's voice. Get the\n"
            "speaker's written agreement for this specific use first, clone a\n"
            "teammate rather than anyone public, and keep the output off any\n"
            "public repository.",
            file=sys.stderr,
        )
        return 2

    if not os.path.isfile(a.ref):
        print(f"reference not found: {a.ref}", file=sys.stderr)
        return 1
    if a.engine == "cmd" and not a.cmd:
        print("--engine cmd needs --cmd", file=sys.stderr)
        return 1

    guard_output_dir(a.out)
    os.makedirs(a.out, exist_ok=True)

    phrases = DEFAULT_PHRASES
    if a.phrases:
        phrases = []
        with open(a.phrases, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if not line.strip() or line.startswith("#"):
                    continue
                parts = line.split("\t", 1)
                phrases.append(
                    (parts[0].strip(), parts[1].strip()) if len(parts) == 2
                    else ("en", line.strip())
                )

    print(f"engine={a.engine}  ref={a.ref}  {len(phrases)} phrases -> {a.out}")
    version = "unknown"
    try:
        if a.engine == "xtts":
            written = synth_xtts(a.ref, phrases, a.out)
            try:
                from importlib.metadata import version as _v
                version = _v("TTS")
            except Exception:  # noqa: BLE001
                pass
        else:
            written = synth_cmd(a.cmd, phrases, a.out, a.ref)
            version = a.cmd
    except ImportError:
        print(
            "\nThe TTS package is not installed. Either:\n"
            "  pip install TTS\n"
            "or use another engine through the escape hatch, e.g. F5-TTS:\n"
            '  --engine cmd --cmd "f5-tts_infer-cli --ref_audio {ref} '
            '--gen_text \\"{text}\\" --output {out}"',
            file=sys.stderr,
        )
        return 3

    prov = {
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "engine": a.engine,
        "engine_version": version,
        "reference_file": os.path.basename(a.ref),
        "reference_sha256_16": sha256(a.ref),
        "n_clips": len(written),
        "phrases": [{"lang": lg, "text": tx} for lg, tx in phrases],
        "split": "eval_unseen",
        "training_use": "FORBIDDEN -- unseen attack split only",
        "consent": "asserted by operator via --i-have-written-consent",
    }
    with open(os.path.join(a.out, "PROVENANCE.json"), "w", encoding="utf-8") as fh:
        json.dump(prov, fh, ensure_ascii=False, indent=1)

    print(f"\nwrote {len(written)} clips + PROVENANCE.json to {a.out}")
    print("\nNext:")
    print("  1. Listen to them. If a clip is obviously robotic, the attack is "
          "weak and detecting it proves little -- regenerate with more "
          "reference audio (30-60s of clean speech helps most).")
    print("  2. Pass them through the call path before scoring: "
          "python ml/codec_augment.py <dir> <dir>_opus --codecs opus")
    print("  3. Score them: python backend/tools/check_cm_polarity.py "
          "--genuine <real dir> --spoof <dir>_opus")
    print("  4. For the phone demo, copy 2-3 into mobile/assets/clones/ and "
          "update AppConfig.cloneClips.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
