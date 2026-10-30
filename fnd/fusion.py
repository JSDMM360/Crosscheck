"""Fusion: combine branch outputs into one fake probability, with every weight exposed.

    logit_i       = clip(logit(p_i))
    w_i           = base_weight_i * reliability_i          (image weight is raised when AI images are found)
    share_i       = w_i / sum(w)
    z             = scale * sum(share_i * logit_i)  +  ai_boost
    P(fake)       = sigmoid(z / temperature)

Missing branches drop out and the remaining shares renormalise. Confidence shrinks with low evidence
coverage and with disagreement between branches.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .config import FusionConfig
from .schema import BranchResult


def _logit(p: float, clip: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return max(-clip, min(clip, math.log(p / (1 - p))))


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


@dataclass
class FusionResult:
    p_fake: float
    p_fake_without_ai_adjustment: float
    label: str
    confidence: float
    rows: list[dict]
    ai_strength: float
    ai_weight_multiplier: float
    ai_boost: float
    z: float
    coverage: float
    disagreement: float
    confidence_raw: float
    caveats: list[str] = field(default_factory=list)
    summary: list[str] = field(default_factory=list)


def _run(branches: list[BranchResult], cfg: FusionConfig, strength: float, use_ai: bool):
    avail = [b for b in branches if b.available]
    mult = 1 + cfg.ai_weight_gain * strength if use_ai else 1.0
    raw = []
    for b in avail:
        w = cfg.weights.get(b.name, 0.1) * max(b.reliability, 0.05)
        if b.name == "image":
            w *= mult
        raw.append(w)
    total = sum(raw) or 1.0
    rows = []
    for b, w in zip(avail, raw):
        lg = _logit(b.p_fake, cfg.logit_clip)
        share = w / total
        rows.append({"branch": b.name, "title": b.title, "p_fake": b.p_fake, "reliability": b.reliability,
                     "base_weight": cfg.weights.get(b.name, 0.1), "effective_weight": w, "share": share,
                     "logit": lg, "contribution": cfg.scale * share * lg})
    boost = cfg.ai_boost_logit * strength if use_ai else 0.0
    z = sum(r["contribution"] for r in rows) + boost
    return rows, z, boost, mult


def fuse(branches: list[BranchResult], cfg: FusionConfig | None = None) -> FusionResult:
    cfg = cfg or FusionConfig()
    img = next((b for b in branches if b.name == "image" and b.available), None)
    strength = float(img.signals.get("ai_strength", 0.0)) if img else 0.0

    rows, z, boost, mult = _run(branches, cfg, strength, True)
    _, z0, _, _ = _run(branches, cfg, strength, False)
    if not rows:
        return FusionResult(0.5, 0.5, "uncertain", 0.0, [], 0, 1, 0, 0, 0, 0, 0,
                            caveats=["No evidence branch could run."])
    t = cfg.temperature
    p = _sigmoid(z / t)
    p0 = _sigmoid(z0 / t)

    avail_names = {r["branch"] for r in rows}
    coverage = sum(cfg.weights.get(n, 0.1) for n in avail_names) / sum(cfg.weights.values())
    mean = sum(r["share"] * r["p_fake"] for r in rows)
    disagreement = math.sqrt(sum(r["share"] * (r["p_fake"] - mean) ** 2 for r in rows))

    raw_conf = max(p, 1 - p)
    conf = 0.5 + (raw_conf - 0.5) * (0.55 + 0.45 * coverage)
    conf -= min(0.15, max(0.0, disagreement - 0.25) * 0.6)
    if len(rows) == 1:
        conf = min(conf, 0.75)
    conf = max(0.5, min(0.99, conf))

    if p >= cfg.fake_threshold and conf >= cfg.min_label_confidence:
        label = "likely fake"
    elif p <= cfg.real_threshold and conf >= cfg.min_label_confidence:
        label = "likely real"
    else:
        label = "uncertain"

    caveats: list[str] = []
    missing = [n for n in cfg.weights if n not in avail_names]
    if missing:
        caveats.append("Branches that did not run: " + ", ".join(missing) + ". Confidence is reduced accordingly.")
    if len(rows) == 1:
        caveats.append(f"Only the '{rows[0]['branch']}' branch ran, so this verdict rests on a single kind of evidence.")
    if disagreement > 0.25:
        caveats.append("The branches disagree strongly. Treat the result as a prompt for manual checking.")
    if strength > 0:
        caveats.append("AI-generated imagery raises risk but does not prove a story false: publishers also use "
                       "AI illustrations openly.")

    summary = []
    for r in sorted(rows, key=lambda r: -abs(r["contribution"]))[:3]:
        direction = "toward fake" if r["contribution"] > 0 else "toward real"
        summary.append(f"{r['title']} pushed {direction} ({r['contribution']:+.2f}, {r['share']*100:.0f}% of weight).")
    if boost > 0:
        summary.append(f"AI-generated image adjustment added {boost:+.2f} and raised the image weight "
                       f"x{mult:.2f}, moving P(fake) from {p0*100:.0f}% to {p*100:.0f}%.")
    return FusionResult(p, p0, label, conf, rows, strength, mult, boost, z, coverage, disagreement,
                        raw_conf, caveats, summary)
