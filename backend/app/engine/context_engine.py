"""
Contextual risk -- financial-urgency keyword spotting.

Two corrections to the usual spec for this component:

1. `faster-whisper` is CTranslate2, not ONNX. "Faster-Whisper-Tiny (ONNX)" is not
   a thing. Pick one: faster-whisper (CT2, easiest, what this module uses) or
   optimum-exported Whisper (real ONNX). Do not claim both.

2. Do not run ASR on the 2.0s CM window. Whisper-tiny hallucinates on sub-3s
   audio, and on Indic input it invents plausible-sounding text. This runs on an
   8s rolling buffer every ~3s, on its own asyncio task, so it never blocks the
   acoustic hot path and never gates the verdict.

The keyword bank is a *corroborating* signal only. In risk_aggregator.py it can
add points but can never raise an alarm on its own -- an urgent-sounding genuine
call must stay green.
"""
from __future__ import annotations

import logging
import os
import re
from typing import List, Optional, Tuple

import numpy as np

from ..core import config as C

log = logging.getLogger(__name__)


# Devanagari / Kannada script, common Latin transliterations, and Indian English.
URGENCY_TERMS = {
    "en": [
        "otp", "one time password", "wire transfer", "transfer the money",
        "immediately", "right now", "urgent", "emergency", "authorise",
        "authorize", "lakh", "crore", "rtgs", "neft", "imps", "upi",
        "account number", "ifsc", "do not tell", "don't tell anyone",
        "keep this confidential", "board approval", "vendor payment",
        "gift card", "aadhaar", "pan card", "cvv", "expiry date",
        "block your account", "kyc", "verification code",
    ],
    "hi": [
        "तुरंत", "अभी", "जल्दी", "पैसे भेजो", "पैसा", "खाता", "खाता नंबर",
        "ओटीपी", "लाख", "करोड़", "आपातकाल", "किसी को मत बताना", "ट्रांसफर",
        "turant", "abhi", "jaldi", "paise bhejo", "khata", "bhejo",
    ],
    "kn": [
        "ತಕ್ಷಣ", "ಈಗಲೇ", "ಹಣ", "ಖಾತೆ", "ಕಳುಹಿಸಿ", "ಲಕ್ಷ", "ತುರ್ತು",
        "takshana", "eegale", "hana", "khate", "kaluhisi", "laksha",
    ],
}

# Word-boundary match for Latin script; substring for Indic scripts, which do not
# tokenise on \b reliably.
_LATIN = re.compile(r"^[\x00-\x7F]+$")


def _compile_bank() -> List[Tuple[str, re.Pattern]]:
    out = []
    for terms in URGENCY_TERMS.values():
        for t in terms:
            if _LATIN.match(t):
                out.append((t, re.compile(r"\b" + re.escape(t) + r"\b", re.I)))
            else:
                out.append((t, re.compile(re.escape(t))))
    return out


_BANK = _compile_bank()


def match_terms(text: str) -> List[str]:
    if not text:
        return []
    return [term for term, pat in _BANK if pat.search(text)]


class ContextEngine:
    """Wraps faster-whisper. Degrades to a no-op if the package is absent."""

    def __init__(self, model_size: str = "tiny") -> None:
        self.available = False
        self._model = None
        self.last_text: str = ""
        self.last_terms: List[str] = []
        try:
            from faster_whisper import WhisperModel

            device = "cpu"
            compute_type = "int8"
            try:
                import torch

                if torch.cuda.is_available():
                    device, compute_type = "cuda", "float16"
            except Exception:  # noqa: BLE001
                pass

            self._model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
                download_root=os.path.join(C.MODELS_DIR, "whisper"),
            )
            self.available = True
            log.info("faster-whisper '%s' ready on %s/%s",
                     model_size, device, compute_type)
        except Exception as exc:  # noqa: BLE001
            log.warning("Context engine disabled (%s)", exc)

    # ------------------------------------------------------------------
    def transcribe(self, audio: np.ndarray, language: Optional[str] = None) -> str:
        if not self.available or audio.size < C.SAMPLE_RATE:
            return ""
        try:
            segments, _info = self._model.transcribe(
                audio.astype(np.float32),
                language=language,          # None => autodetect (hi/kn/en)
                beam_size=1,                # greedy: this is a latency budget, not a benchmark
                vad_filter=False,           # we already gate upstream
                condition_on_previous_text=False,  # stops runaway hallucination loops
            )
            return " ".join(s.text for s in segments).strip()
        except Exception as exc:  # noqa: BLE001
            log.exception("ASR failed: %s", exc)
            return ""

    def analyse(self, audio: np.ndarray) -> Tuple[bool, List[str], str]:
        text = self.transcribe(audio)
        terms = match_terms(text)
        self.last_text = text
        self.last_terms = terms
        return bool(terms), terms, text
