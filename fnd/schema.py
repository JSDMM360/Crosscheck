from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class BranchResult:
    """Output of one modality branch, in a shape the fusion layer can consume."""
    name: str
    title: str
    available: bool
    p_fake: float = 0.5          # branch-level probability that the input is fake
    reliability: float = 1.0     # 0..1: how much evidence the branch actually had
    signals: dict[str, Any] = field(default_factory=dict)
    evidence: list[dict[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @classmethod
    def unavailable(cls, name: str, title: str, note: str) -> "BranchResult":
        return cls(name=name, title=title, available=False, p_fake=0.5,
                   reliability=0.0, notes=[note])


@dataclass
class ImageSignal:
    name: str
    p_ai: Optional[float]        # None = informational only, not used in scoring
    weight: float
    detail: str
    decisive: bool = False       # explicit provenance declaring AI generation


@dataclass
class ImageReport:
    source: str
    width: int = 0
    height: int = 0
    p_ai: float = 0.0
    verdict: str = "inconclusive"
    reliability: float = 0.0
    signals: list[ImageSignal] = field(default_factory=list)
    disclosed: bool = False      # the page itself labels the image as AI-made
    alt: str = ""
    thumbnail: Optional[bytes] = None
    error: Optional[str] = None
