from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"


@dataclass
class Settings:
    # acquisition
    timeout: int = 12
    max_images: int = 4
    fetch_linked_pages: bool = False
    max_linked_pages: int = 3
    user_agent: str = "Mozilla/5.0 (compatible; CrosscheckBot/1.0; +local-research-tool)"
    # image detection
    use_hf_image_model: bool = os.getenv("FND_DISABLE_HF", "0") != "1"
    image_models: list[str] = field(default_factory=lambda: [
        m.strip() for m in os.getenv("FND_IMAGE_MODELS", "umm-maybe/AI-image-detector").split(",") if m.strip()
    ])
    # optional local LLM (Ollama) for plausibility + plain-language explanation
    use_ollama: bool = False
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2")
    # evidence retrieval
    news_region: str = os.getenv("FND_NEWS_REGION", "IN")
    news_lang: str = os.getenv("FND_NEWS_LANG", "en")
    factcheck_api_key: str = os.getenv("GOOGLE_FACTCHECK_API_KEY", "")


@dataclass
class FusionConfig:
    # Base weights per branch. These are hand-set priors, NOT learned values.
    weights: dict[str, float] = field(default_factory=lambda: {
        "text": 0.22, "verifiability": 0.10, "source": 0.16, "link": 0.10,
        "corroboration": 0.12, "factcheck": 0.14, "image": 0.16,
    })
    scale: float = 1.2             # converts the weighted mean logit into the final logit
    temperature: float = 1.0       # >1 softens, <1 sharpens (fit on validation data when available)
    logit_clip: float = 3.0
    # AI-generated image handling
    ai_flag_threshold: float = 0.60   # image counts as "generated" at or above this P(AI)
    ai_weight_gain: float = 2.0       # image weight is multiplied by (1 + gain * strength)
    ai_boost_logit: float = 1.2       # extra push toward fake at strength 1.0
    ai_disclosed_discount: float = 0.4  # strength multiplier if the page openly labels the image as AI
    fake_threshold: float = 0.65
    real_threshold: float = 0.35
    min_label_confidence: float = 0.60
