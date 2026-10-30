"""Text branches: writing style (headline + body) and claim verifiability.

Two layers feed the style score:
  * a transparent heuristic (always on, uncalibrated), and
  * an optional trained TF-IDF + logistic regression model at models/text_clf.joblib
    (create it with train_text_model.py). Only load model files you trained yourself.
Verifiability can also use a local LLM through Ollama (optional).
"""
from __future__ import annotations

import math
import re
from typing import Optional

import requests

from .config import MODELS_DIR, Settings
from .schema import BranchResult

CLICKBAIT = [
    r"you won'?t believe", r"what happens next", r"doctors hate", r"this one (trick|weird)", r"goes viral",
    r"must[- ]see", r"they don'?t want you to know", r"share (this )?before", r"\bexposed\b", r"\bbombshell\b",
    r"miracle", r"100% (proven|guaranteed)", r"shocking", r"mind[- ]blowing", r"what (they|he|she) did next",
    r"breaking[:!]", r"urgent[:!]", r"wake up",
]
SENSATIONAL = set("""outrage outraged destroyed slams slammed horrifying unbelievable insane terrifying explosive
devastating disgusting traitor evil hoax conspiracy secret leaked banned cover-up coverup scandal miracle
deadly massive epic savage""".split())
SUPERLATIVES = set("""largest highest biggest greatest worst best most least never always everyone nobody
all every entire ever record-breaking unprecedented historic""".split())
ATTRIB = re.compile(r"\b(said|says|told|according to|reported|stated|announced|confirmed|added|spokesperson|statement)\b", re.I)


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def _heuristic(headline: str, body: str) -> tuple[float, dict]:
    head = headline.strip()
    words = re.findall(r"[A-Za-z][A-Za-z'-]+", head)
    low = head.lower()
    click = sum(bool(re.search(p, low)) for p in CLICKBAIT)
    sens = sum(w.lower() in SENSATIONAL for w in re.findall(r"[A-Za-z'-]+", head + " " + body[:1500]))
    caps = sum(1 for w in words if len(w) >= 3 and w.isupper())
    caps_ratio = caps / max(1, len(words))
    excl = head.count("!") + body[:1500].count("!!")
    toks = [w.lower() for w in words]
    sup = sum(t in SUPERLATIVES for t in toks)
    double_sup = sum(1 for a, b in zip(toks, toks[1:]) if a in SUPERLATIVES and b in SUPERLATIVES)
    nbody = len(body.split())
    no_attrib = nbody >= 80 and not ATTRIB.search(body)

    z = (-0.6 + 0.9 * min(click, 3) + 0.5 * min(sens, 4) + 2.0 * caps_ratio + 0.5 * min(excl, 3)
         + 0.35 * min(sup, 3) + 1.2 * min(double_sup, 2) + (0.7 if no_attrib else 0.0)
         + (0.3 if head.endswith("?") else 0.0))
    sig = {"clickbait_phrases": click, "sensational_words": sens, "all_caps_ratio": round(caps_ratio, 2),
           "exclamations": excl, "superlatives": sup, "stacked_superlatives": double_sup,
           "body_has_attribution": not no_attrib if nbody >= 80 else "n/a (short text)"}
    return _sigmoid(z), sig


_CLF = {"loaded": False, "obj": None}


def _trained_model() -> Optional[object]:
    if not _CLF["loaded"]:
        _CLF["loaded"] = True
        path = MODELS_DIR / "text_clf.joblib"
        if path.exists():
            try:
                import joblib
                _CLF["obj"] = joblib.load(path)
            except Exception:
                _CLF["obj"] = None
    return _CLF["obj"]


def text_branch(headline: str, body: str) -> BranchResult:
    if not (headline or body).strip():
        return BranchResult.unavailable("text", "Writing style", "No text provided.")
    p_h, sig = _heuristic(headline, body)
    notes = []
    model = _trained_model()
    p, rel = p_h, 0.55 if len(body.split()) < 80 else 0.7
    if model is not None:
        try:
            p_m = float(model.predict_proba([f"{headline}. {body[:4000]}"])[0][1])
            sig["trained_model_p_fake"] = round(p_m, 3)
            p, rel = 0.7 * p_m + 0.3 * p_h, 0.85
        except Exception as e:
            notes.append(f"Trained model failed ({e.__class__.__name__}); heuristic used.")
    else:
        notes.append("Using the built-in style heuristic (uncalibrated). Train models/text_clf.joblib "
                     "with train_text_model.py for a data-driven text score.")
    sig["heuristic_p_fake"] = round(p_h, 3)
    return BranchResult("text", "Writing style", True, p, rel, sig, notes=notes)


# ------------------------------------------------------------------ verifiability
_ENT = re.compile(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*)\b")
_NUM = re.compile(r"\b(\d{1,4}(?:[.,]\d+)?%?|(?:19|20)\d{2})\b")
_QUOTE = re.compile(r"[\"“”][^\"“”]{15,}[\"“”]")


def ollama_plausibility(claim: str, body: str, settings: Settings) -> Optional[tuple[float, str]]:
    prompt = (
        "You are a careful fact-checking assistant. Judge ONLY whether the claim is internally coherent and "
        "plausible given general world knowledge. Do not assume it is true or false.\n"
        f"Claim: {claim}\nContext: {body[:1200]}\n"
        'Reply with JSON only: {"implausibility": <0.0-1.0>, "reason": "<one sentence>"}'
    )
    try:
        r = requests.post(f"{settings.ollama_url}/api/generate", timeout=60,
                          json={"model": settings.ollama_model, "prompt": prompt, "stream": False, "format": "json"})
        r.raise_for_status()
        import json
        out = json.loads(r.json()["response"])
        return min(1.0, max(0.0, float(out["implausibility"]))), str(out.get("reason", ""))[:300]
    except Exception:
        return None


def verifiability_branch(headline: str, body: str, settings: Settings) -> BranchResult:
    claim = headline.strip() or body[:200]
    if not claim:
        return BranchResult.unavailable("verifiability", "Claim verifiability", "No claim text.")
    text = f"{headline}. {body[:1500]}"
    ents = {m.group(1) for m in _ENT.finditer(text)}
    ents |= {w for w in re.findall(r"\b[A-Z]{2,6}\b", text) if w not in {"AI", "THE", "AND", "NOT"}}
    nums = _NUM.findall(text)
    quotes = len(_QUOTE.findall(body))
    verifiable = min(len(ents), 5) + min(len(nums), 3) + min(quotes, 2)
    z = 0.9 - 0.3 * verifiable                       # fewer checkable specifics -> more suspicious
    sig = {"named_entities": len(ents), "numbers_or_dates": len(nums), "attributed_quotes": quotes,
           "verifiable_specifics": verifiable}
    notes, rel = [], 0.45
    if settings.use_ollama:
        res = ollama_plausibility(claim, body, settings)
        if res:
            imp, reason = res
            z = 0.4 * z + 0.6 * (4 * imp - 2)
            sig["llm_implausibility"] = round(imp, 2)
            rel = 0.65
            notes.append(f"Local LLM ({settings.ollama_model}): {reason}")
        else:
            notes.append("Ollama was enabled but did not answer. Is `ollama serve` running?")
    else:
        notes.append("Counts checkable specifics only. It cannot judge whether a claim makes sense. "
                     "Enable the Ollama option to add a plausibility check.")
    return BranchResult("verifiability", "Claim verifiability", True, _sigmoid(z), rel, sig, notes=notes)
