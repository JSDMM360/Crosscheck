"""Source/metadata, link-context, cross-outlet corroboration and fact-check branches."""
from __future__ import annotations

import difflib
import math
import re
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus, urlparse

import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .acquisition import FetchError, Page, base_domain, fetch_text, load_reputation, parse_page
from .config import Settings
from .schema import BranchResult


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def _tier(host: str, rep: dict) -> str:
    d = base_domain(host)
    if d in rep["credible"]:
        return "credible"
    if d in rep["satire"]:
        return "satire"
    if d in rep["unreliable"]:
        return "unreliable"
    return "unknown"


# ------------------------------------------------------------------ source
def source_branch(page: Page) -> BranchResult:
    rep = load_reputation()
    host = urlparse(page.final_url).hostname or ""
    dom = base_domain(host)
    tier = _tier(host, rep)
    z, notes = 0.0, []
    sig = {"domain": dom, "reputation_tier": tier, "https": page.https,
           "byline_present": bool(page.author), "date_present": bool(page.date), "ad_elements": page.ad_count}
    if tier == "credible":
        z -= 2.2
    elif tier == "satire":
        z += 1.5
        notes.append("This domain is on the satire list: the content is probably satire, not deliberate disinformation.")
    elif tier == "unreliable":
        z += 1.8
    if not page.https:
        z += 0.4
    tld = dom.rsplit(".", 1)[-1]
    if tld in rep["suspicious_tlds"]:
        z += 0.8
        notes.append(f".{tld} is a top-level domain often used by throwaway sites.")
    if tier == "unknown":
        for good in rep["credible"]:
            r = difflib.SequenceMatcher(None, dom, good).ratio()
            if r >= 0.85 and dom != good:
                z += 1.8
                sig["possible_impersonation_of"] = good
                notes.append(f"The domain closely resembles {good}, which can indicate impersonation.")
                break
    z += -0.3 if page.author else 0.5
    z += -0.2 if page.date else 0.4
    if page.ad_count > 8:
        z += 0.5
    rel = 0.9 if tier != "unknown" else 0.5
    if tier == "unknown":
        notes.append("Domain is not in the reputation list, so only page-level signals were used.")
    return BranchResult("source", "Source & page metadata", True, _sigmoid(z), rel, sig, notes=notes)


# ------------------------------------------------------------------ links
def link_branch(page: Page, settings: Settings) -> BranchResult:
    rep = load_reputation()
    editorial = [l for l in page.links if l.kind == "editorial"]
    counts = {k: sum(l.kind == k for l in page.links) for k in ("editorial", "internal", "navigation", "advertising")}
    sig = {"link_counts": counts, "raw_link_count": len(page.links)}
    if page.word_count < 150 and not editorial:
        return BranchResult.unavailable("link", "Link context", "Page too short or has no editorial links to judge.")
    if not editorial:
        sig.update(relevant_link_ratio=None, credible_source_share=None)
        return BranchResult("link", "Link context", True, 0.58, 0.40, sig,
                            notes=["The article cites no external sources, so its claims cannot be traced."])

    tiers = [_tier(urlparse(l.url).hostname or "", rep) for l in editorial]
    credible_share = tiers.count("credible") / len(editorial)
    bad_share = (tiers.count("satire") + tiers.count("unreliable")) / len(editorial)

    # topical relevance: anchor + surrounding sentence + URL slug vs the page text
    docs = [f"{l.anchor} {l.context} {re.sub(r'[-_/]', ' ', urlparse(l.url).path)}" for l in editorial]
    notes, rel = [], 0.5
    if settings.fetch_linked_pages:
        for i, l in enumerate(editorial[: settings.max_linked_pages]):
            try:
                html, final = fetch_text(l.url, settings)
                docs[i] = parse_page(html, final).text[:3000] or docs[i]
                rel = 0.65
            except FetchError as e:
                notes.append(f"Could not fetch {l.url}: {e}")
    try:
        vec = TfidfVectorizer(stop_words="english").fit([page.text] + docs)
        sims = cosine_similarity(vec.transform([page.text]), vec.transform(docs))[0]
        ratio = float((sims >= 0.08).mean())
    except ValueError:
        ratio = 0.0
    sig.update(relevant_link_ratio=round(ratio, 2), credible_source_share=round(credible_share, 2),
               unreliable_or_satire_share=round(bad_share, 2),
               linked_pages_fetched=min(len(editorial), settings.max_linked_pages) if settings.fetch_linked_pages else 0)
    z = 0.3 - 1.6 * credible_share - 0.8 * ratio + 1.0 * bad_share
    if not settings.fetch_linked_pages:
        notes.append("Relevance is judged from anchor text and URL only. Enable 'fetch linked pages' to compare page content.")
    return BranchResult("link", "Link context", True, _sigmoid(z), rel, sig,
                        evidence=[{"title": l.anchor[:80] or l.url, "detail": f"{t}: {l.url}"} for l, t in zip(editorial[:8], tiers)],
                        notes=notes)


# ------------------------------------------------------------------ corroboration
_STOP = set("a an the of in on at to for by with from and or as is are was were be been that this it its has have had".split())


def _query(headline: str) -> str:
    words = [w for w in re.findall(r"[A-Za-z0-9'-]+", headline) if w.lower() not in _STOP]
    return " ".join(words[:9])


def corroboration_branch(headline: str, settings: Settings, rss_text: str | None = None) -> BranchResult:
    title = "Cross-outlet corroboration"
    q = _query(headline)
    if not q:
        return BranchResult.unavailable("corroboration", title, "No headline to search for.")
    if rss_text is None:
        url = (f"https://news.google.com/rss/search?q={quote_plus(q)}"
               f"&hl={settings.news_lang}-{settings.news_region}&gl={settings.news_region}&ceid={settings.news_region}:{settings.news_lang}")
        try:
            r = requests.get(url, timeout=settings.timeout, headers={"User-Agent": settings.user_agent})
            r.raise_for_status()
            rss_text = r.text
        except requests.RequestException as e:
            return BranchResult.unavailable("corroboration", title, f"News search unreachable ({e.__class__.__name__}).")
    try:
        items = ET.fromstring(rss_text).findall(".//item")
    except ET.ParseError:
        return BranchResult.unavailable("corroboration", title, "News search returned unreadable data.")

    rep = load_reputation()
    cands = []
    for it in items[:30]:
        t = (it.findtext("title") or "").rsplit(" - ", 1)[0]
        src = it.find("source")
        host = urlparse(src.get("url", "")).hostname if src is not None and src.get("url") else ""
        cands.append({"title": t, "outlet": (src.text if src is not None else "") or host or "?",
                      "host": host or "", "link": it.findtext("link") or ""})
    sims = []
    if cands:
        vec = TfidfVectorizer(stop_words="english").fit([headline] + [c["title"] for c in cands])
        sims = cosine_similarity(vec.transform([headline]), vec.transform([c["title"] for c in cands]))[0]
    seen_outlets: set[str] = set()
    matched, score = [], 0.0
    for c, s in sorted(zip(cands, sims), key=lambda x: -x[1]):
        key = base_domain(c["host"]) or c["outlet"].lower()
        if s < 0.35 or key in seen_outlets:         # similarity floor + de-duplicate per outlet
            continue
        seen_outlets.add(key)
        tier = _tier(c["host"], rep) if c["host"] else "unknown"
        w = {"credible": 1.0, "unknown": 0.35, "unreliable": 0.1, "satire": 0.0}[tier]
        score += w * float(s)
        matched.append({"title": c["title"][:110], "detail": f"{c['outlet']} ({tier}, similarity {s:.2f}) {c['link']}"})
    strength = 1 - math.exp(-1.5 * score)
    z = max(-1.5, min(0.6, 0.5 - 2.0 * strength))   # bounded: never decides the verdict alone
    notes = [] if matched else ["No matching coverage found. The story may be new or niche, so this counts only mildly against it."]
    return BranchResult("corroboration", title, True, _sigmoid(z), 0.7,
                        {"query": q, "candidates_checked": len(cands), "independent_matches": len(matched),
                         "corroboration_score": round(strength, 3)}, evidence=matched[:8], notes=notes)


# ------------------------------------------------------------------ fact-check records
_FALSE = re.compile(r"false|fake|incorrect|misleading|pants on fire|fabricated|hoax|unproven|no evidence|satire", re.I)
_TRUE = re.compile(r"\btrue\b|correct|accurate|\bfact\b|legit", re.I)


def factcheck_branch(headline: str, settings: Settings) -> BranchResult:
    title = "Fact-check records"
    if not settings.factcheck_api_key:
        return BranchResult.unavailable("factcheck", title, "Set GOOGLE_FACTCHECK_API_KEY to query fact-check databases.")
    try:
        r = requests.get("https://factchecktools.googleapis.com/v1alpha1/claims:search", timeout=settings.timeout,
                         params={"query": headline[:200], "key": settings.factcheck_api_key, "pageSize": 10})
        r.raise_for_status()
        claims = r.json().get("claims", [])
    except (requests.RequestException, ValueError) as e:
        return BranchResult.unavailable("factcheck", title, f"Fact-check lookup failed ({e.__class__.__name__}).")
    fake = real = 0
    ev = []
    for c in claims:
        for rv in c.get("claimReview", []):
            rating = rv.get("textualRating", "")
            fake += bool(_FALSE.search(rating))
            real += bool(_TRUE.search(rating)) and not _FALSE.search(rating)
            ev.append({"title": (rv.get("title") or c.get("text", ""))[:110],
                       "detail": f"{rv.get('publisher', {}).get('name', '?')}: {rating} {rv.get('url', '')}"})
    if fake + real == 0:
        return BranchResult.unavailable("factcheck", title, "No matching fact-check entry found.")
    p = (fake + 0.5) / (fake + real + 1)
    return BranchResult("factcheck", title, True, p, 0.8, {"fake_ratings": fake, "true_ratings": real}, evidence=ev[:8])
