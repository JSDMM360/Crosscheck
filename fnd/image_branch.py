"""Image branch: was this image generated (Gemini, DALL-E, Midjourney, Stable Diffusion...) or captured?

Evidence layers, strongest first:
  1. Provenance metadata: IPTC DigitalSourceType, C2PA manifests, generator names in EXIF/XMP/PNG text,
     Stable Diffusion / ComfyUI parameter chunks. Decisive when present, but easily stripped.
  2. A learned detector (Hugging Face image-classification model, optional).
  3. Weak heuristics (typical generator resolutions, missing camera metadata, noise uniformity).
Invisible watermarks such as Google's SynthID cannot be read without the vendor's own detector.
"""
from __future__ import annotations

import io
import math
import re
from typing import Optional

import numpy as np
from PIL import Image, ImageFilter

from .config import FusionConfig, Settings
from .schema import BranchResult, ImageReport, ImageSignal

PRIOR_LOGIT = -0.6      # most news photos are real; this is the starting belief before any signal

_GEN_PATTERNS = [
    r"dall[-·. ]?e", r"midjourney", r"stable[ -]?diffusion", r"\bsdxl\b", r"comfyui", r"automatic1111",
    r"novelai", r"adobe firefly", r"\bgemini\b", r"google imagen", r"\bimagen[ -]?\d\b", r"gpt-image",
    r"chatgpt", r"openai", r"leonardo\.ai", r"ideogram", r"flux\.1", r"black forest labs", r"dreamstudio",
    r"craiyon", r"nightcafe", r"bing image creator", r"\bgrok\b", r"made with google ai", r"synthid",
    r"ai[- ]generated", r"generated (by|with|using) ai",
]
_GEN_RE = re.compile("|".join(_GEN_PATTERNS), re.I)
_COMMON_AI_SIZES = {
    (1024, 1024), (1024, 1792), (1792, 1024), (1536, 1024), (1024, 1536), (768, 768), (512, 512),
    (2048, 2048), (1344, 768), (768, 1344), (1216, 832), (832, 1216), (1152, 896), (896, 1152),
    (1408, 768), (1280, 720), (1456, 816),
}


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


# --------------------------------------------------------------------------- provenance
def _metadata_text(data: bytes, img: Image.Image) -> str:
    parts: list[str] = []
    try:
        exif = img.getexif()
        for tag in (0x0131, 0x010E, 0x013B, 0x8298, 0x010F, 0x0110):
            v = exif.get(tag)
            if isinstance(v, bytes):
                v = v.decode("utf-8", "ignore")
            if v:
                parts.append(str(v))
        try:
            uc = exif.get_ifd(0x8769).get(0x9286)
            if isinstance(uc, bytes):
                parts.append(uc.decode("utf-8", "ignore"))
        except Exception:
            pass
    except Exception:
        pass
    for k, v in (img.info or {}).items():
        if isinstance(v, bytes):
            continue
        if isinstance(v, str):
            parts.append(f"{k}: {v[:5000]}")
    m = re.search(rb"<x:xmpmeta.*?</x:xmpmeta>", data, re.S)
    if m:
        parts.append(m.group(0).decode("utf-8", "ignore"))
    return "\n".join(parts)


def _provenance(data: bytes, img: Image.Image) -> list[ImageSignal]:
    sigs: list[ImageSignal] = []
    low = data.lower()

    if b"trainedalgorithmicmedia" in low:
        sigs.append(ImageSignal("IPTC digital source type", 0.98, 1.0,
                                "Metadata declares the image as trainedAlgorithmicMedia (AI-generated).", True))
    elif b"compositesynthetic" in low:
        sigs.append(ImageSignal("IPTC digital source type", 0.90, 1.0,
                                "Metadata declares the image as composite synthetic media.", True))

    c2pa_at = min([i for i in (low.find(b"c2pa"), low.find(b"jumb")) if i >= 0], default=-1)
    if c2pa_at >= 0:
        region = data[c2pa_at:c2pa_at + 400_000].decode("latin-1", "ignore")
        hit = _GEN_RE.search(region)
        if hit:
            sigs.append(ImageSignal("C2PA content credentials", 0.96, 1.0,
                                    f"Manifest names an AI generator ('{hit.group(0)}').", True))
        else:
            sigs.append(ImageSignal("C2PA content credentials", 0.25, 0.5,
                                    "A C2PA manifest is present and names no AI generator. "
                                    "That shows provenance exists, not that the content is true."))

    info = img.info or {}
    params = str(info.get("parameters", "")).lower()
    if "steps:" in params and ("sampler" in params or "cfg scale" in params):
        sigs.append(ImageSignal("Generation parameters", 0.97, 1.0,
                                "PNG text chunk holds Stable Diffusion style parameters (steps, sampler, CFG).", True))
    elif "prompt" in info and "workflow" in info:
        sigs.append(ImageSignal("Generation workflow", 0.97, 1.0,
                                "PNG holds a ComfyUI prompt/workflow graph.", True))

    if not any(s.decisive for s in sigs):
        hit = _GEN_RE.search(_metadata_text(data, img))
        if hit:
            sigs.append(ImageSignal("Generator named in metadata", 0.93, 1.0,
                                    f"EXIF/XMP/PNG text mentions '{hit.group(0)}'.", True))
    return sigs


def _camera_signal(img: Image.Image) -> Optional[ImageSignal]:
    try:
        exif = img.getexif()
        sub = exif.get_ifd(0x8769)
        has_cam = bool(exif.get(0x010F) and exif.get(0x0110))
        has_shot = any(k in sub for k in (0x829A, 0x829D, 0x8827, 0x9003))
        if has_cam and has_shot:
            return ImageSignal("Camera capture metadata", 0.12, 0.7,
                               f"Make/model and exposure data present ({exif.get(0x010F)} {exif.get(0x0110)}). "
                               "Editable, so this is only mild evidence of a real capture.")
    except Exception:
        pass
    return ImageSignal("No camera metadata", 0.60, 0.3,
                       "No camera make/model or exposure data. Most web images are stripped, so this is weak.")


# --------------------------------------------------------------------------- heuristics
def _forensics(img: Image.Image) -> list[ImageSignal]:
    sigs: list[ImageSignal] = []
    g = img.convert("L")
    if max(g.size) > 1024:
        s = 1024 / max(g.size)
        g = g.resize((max(1, int(g.width * s)), max(1, int(g.height * s))))
    arr = np.asarray(g, dtype=np.float32)
    resid = arr - np.asarray(g.filter(ImageFilter.GaussianBlur(1.0)), dtype=np.float32)
    noise = float(resid.std())
    bs = 32
    h, w = (resid.shape[0] // bs) * bs, (resid.shape[1] // bs) * bs
    cv = 0.0
    if h >= bs and w >= bs:
        blocks = resid[:h, :w].reshape(h // bs, bs, w // bs, bs).std(axis=(1, 3))
        cv = float(blocks.std() / (blocks.mean() + 1e-6))
    smooth = noise < 1.0 and cv < 0.6
    sigs.append(ImageSignal(
        "Noise uniformity", 0.62 if smooth else 0.5, 0.25,
        f"Residual noise std {noise:.2f}, block variation {cv:.2f}. "
        + ("Unusually smooth and uniform, which generated images often show (resized or compressed photos can too)."
           if smooth else "Nothing unusual. This heuristic is uncalibrated and weak.")))
    centred = resid - resid.mean()
    spec = np.abs(np.fft.fftshift(np.fft.fft2(centred)))
    cy, cx = np.array(spec.shape) // 2
    spec[max(0, cy - 4):cy + 5, max(0, cx - 4):cx + 5] = 0
    ratio = float(spec.max() / (np.median(spec) + 1e-6))
    sigs.append(ImageSignal("Spectral peak ratio", None, 0.0,
                            f"{ratio:.0f}. Shown for inspection only: periodic upsampling artefacts raise it, "
                            "but it is not calibrated, so it does not affect the score."))
    return sigs


_PIPES: dict = {}
_AI_WORDS = ("artificial", "ai", "fake", "generated", "synthetic", "sdxl")
_REAL_WORDS = ("human", "real", "natural", "authentic")


def _hf_signal(img: Image.Image, settings: Settings) -> Optional[ImageSignal]:
    if not settings.use_hf_image_model or not settings.image_models:
        return None
    try:
        from transformers import pipeline
    except Exception:
        return ImageSignal("Learned detector", None, 0.0,
                           "Not run: install `torch` and `transformers` to enable the learned detector.")
    probs, used = [], []
    for mid in settings.image_models:
        try:
            if mid not in _PIPES:
                _PIPES[mid] = pipeline("image-classification", model=mid)
            out = _PIPES[mid](img.convert("RGB"), top_k=None)
            p_ai = None
            for o in out:
                lab = o["label"].lower()
                if any(w in lab for w in _AI_WORDS):
                    p_ai = o["score"]
                    break
                if any(w in lab for w in _REAL_WORDS):
                    p_ai = 1 - o["score"]
            if p_ai is not None:
                probs.append(p_ai)
                used.append(f"{mid}: {p_ai:.2f}")
        except Exception as e:
            used.append(f"{mid}: unavailable ({e.__class__.__name__})")
    if not probs:
        return ImageSignal("Learned detector", None, 0.0, "Model could not be loaded. " + "; ".join(used))
    return ImageSignal("Learned detector", float(np.mean(probs)), 1.0,
                       "P(AI) from " + "; ".join(used) + ". Public detectors age quickly against new generators "
                       "and degrade on resized or recompressed images.")


# --------------------------------------------------------------------------- per image
def analyze_image(data: bytes, source: str, settings: Settings,
                  alt: str = "", caption: str = "") -> ImageReport:
    rep = ImageReport(source=source, alt=alt)
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as e:
        rep.error = f"Could not decode image ({e.__class__.__name__})."
        return rep
    rep.width, rep.height = img.size
    try:
        t = img.convert("RGB")
        t.thumbnail((360, 360))
        buf = io.BytesIO()
        t.save(buf, "JPEG", quality=80)
        rep.thumbnail = buf.getvalue()
    except Exception:
        pass

    sigs = _provenance(data, img)
    cam = _camera_signal(img)
    if cam:
        sigs.append(cam)
    if (rep.width, rep.height) in _COMMON_AI_SIZES:
        sigs.append(ImageSignal("Typical generator resolution", 0.62, 0.3,
                                f"{rep.width}x{rep.height} is a common generator output size."))
    sigs += _forensics(img)
    hf = _hf_signal(img, settings)
    if hf:
        sigs.append(hf)
    rep.signals = sigs
    rep.disclosed = bool(re.search(r"ai[- ]generated|generated (by|with|using) ai|ai[- ]illustration", f"{alt} {caption}", re.I))

    decisive = [s.p_ai for s in sigs if s.decisive and s.p_ai is not None]
    scored = [s for s in sigs if s.p_ai is not None and s.weight > 0]
    has_model = any(s.name == "Learned detector" and s.p_ai is not None for s in sigs)
    if decisive:
        rep.p_ai, rep.reliability = max(decisive), 0.95
    else:
        z = PRIOR_LOGIT + sum(s.weight * max(-4, min(4, _logit(s.p_ai))) for s in scored)
        rep.p_ai = _sigmoid(z)
        rep.reliability = 0.75 if has_model else 0.40
    if rep.p_ai >= 0.60:
        rep.verdict = "likely AI-generated"
    elif rep.p_ai <= 0.30 and rep.reliability >= 0.5:
        rep.verdict = "likely authentic"
    else:
        rep.verdict = "inconclusive"
    return rep


# --------------------------------------------------------------------------- branch
def aggregate(reports: list[ImageReport], page_discloses: bool, cfg: FusionConfig) -> BranchResult:
    valid = [r for r in reports if r.error is None]
    if not valid:
        note = "No decodable images." if reports else "No images were provided or found on the page."
        return BranchResult.unavailable("image", "Image authenticity", note)

    flagged = [r for r in valid if r.p_ai >= cfg.ai_flag_threshold]
    strength, discount = 0.0, 1.0
    if flagged:
        max_conf = max((r.p_ai - 0.5) / 0.5 for r in flagged)
        frac = len(flagged) / len(valid)
        strength = max_conf * (0.7 + 0.3 * frac)
        if page_discloses or all(r.disclosed for r in flagged):
            discount = cfg.ai_disclosed_discount
            strength *= discount
    p_fake = 0.40 + 0.50 * strength if flagged else 0.40
    reliability = sum(r.reliability for r in valid) / len(valid)

    notes = []
    if flagged:
        notes.append(f"{len(flagged)} of {len(valid)} image(s) look AI-generated. "
                     "An AI image is a risk signal, not proof that the story is false.")
        if discount < 1:
            notes.append("The page labels its imagery as AI-made, so the penalty was reduced.")
    if not any(any(s.name == "Learned detector" and s.p_ai is not None for s in r.signals) for r in valid):
        notes.append("No learned detector ran: results rely on provenance metadata and weak heuristics. "
                     "AI images with stripped metadata will often be missed.")
    return BranchResult(
        "image", "Image authenticity", True, p_fake, reliability,
        signals={"images_analysed": len(valid), "images_flagged_ai": len(flagged),
                 "max_p_ai": round(max(r.p_ai for r in valid), 3), "ai_strength": round(strength, 3),
                 "disclosure_discount": discount},
        evidence=[{"title": r.source, "detail": f"{r.verdict} (P(AI) {r.p_ai:.2f})"} for r in valid],
        notes=notes)
