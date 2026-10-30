from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

from . import evidence_branches as ev
from .acquisition import FetchError, Page, fetch_image, fetch_page
from .config import FusionConfig, Settings
from .fusion import FusionResult, fuse
from .image_branch import aggregate, analyze_image
from .schema import BranchResult, ImageReport
from .text_branch import text_branch, verifiability_branch


@dataclass
class Analysis:
    headline: str
    url: str
    page: Optional[Page]
    branches: list[BranchResult]
    images: list[ImageReport]
    fusion: FusionResult
    seconds: float
    notes: list[str] = field(default_factory=list)


def _safe(name: str, title: str, fn) -> BranchResult:
    try:
        return fn()
    except Exception as e:  # a failing branch must never take the whole analysis down
        return BranchResult.unavailable(name, title, f"Branch failed: {e.__class__.__name__}: {e}")


def analyze(headline: str = "", url: str = "", uploaded: list[tuple[str, bytes]] | None = None,
            settings: Settings | None = None, cfg: FusionConfig | None = None) -> Analysis:
    t0 = time.time()
    settings, cfg = settings or Settings(), cfg or FusionConfig()
    headline, url, uploaded = (headline or "").strip(), (url or "").strip(), uploaded or []
    notes: list[str] = []
    page: Optional[Page] = None

    if url:
        try:
            page = fetch_page(url, settings)
        except FetchError as e:
            notes.append(f"Could not read the page: {e}")
    claim = headline or (page.title if page else "")
    body = page.text if page else ""

    # ---- images: uploads + images found on the page
    reports: list[ImageReport] = []
    for name, data in uploaded:
        reports.append(analyze_image(data, f"upload: {name}", settings))
    if page:
        def work(pi):
            try:
                return analyze_image(fetch_image(pi.url, settings), pi.url, settings, pi.alt, pi.caption)
            except FetchError as e:
                return ImageReport(source=pi.url, error=str(e))
        with ThreadPoolExecutor(max_workers=4) as pool:
            reports += list(pool.map(work, page.images[: settings.max_images]))
    page_discloses = bool(page and page.discloses_ai_images)

    # ---- network-bound evidence branches run in parallel
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_corr = pool.submit(lambda: _safe("corroboration", "Cross-outlet corroboration",
                                           lambda: ev.corroboration_branch(claim, settings))) if claim else None
        f_fact = pool.submit(lambda: _safe("factcheck", "Fact-check records",
                                           lambda: ev.factcheck_branch(claim, settings))) if claim else None
        branches = [
            _safe("text", "Writing style", lambda: text_branch(claim, body)),
            _safe("verifiability", "Claim verifiability", lambda: verifiability_branch(claim, body, settings)),
        ]
        if page:
            branches.append(_safe("source", "Source & page metadata", lambda: ev.source_branch(page)))
            branches.append(_safe("link", "Link context", lambda: ev.link_branch(page, settings)))
        branches.append(_safe("image", "Image authenticity", lambda: aggregate(reports, page_discloses, cfg)))
        if f_corr:
            branches.append(f_corr.result())
        if f_fact:
            branches.append(f_fact.result())

    return Analysis(claim, url, page, branches, reports, fuse(branches, cfg), time.time() - t0, notes)
