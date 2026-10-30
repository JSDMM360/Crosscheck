"""Fetch a URL and extract title, text, author, date, images and outbound links."""
from __future__ import annotations

import ipaddress
import json
import os
import re
import socket
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from .config import DATA_DIR, Settings

MAX_HTML = 3_000_000
MAX_IMG = 8_000_000

_SLD = {"co", "com", "org", "net", "gov", "ac", "edu"}

DISCLOSURE_RE = re.compile(
    r"(ai[- ]generated|generated\s+(by|with|using)\s+(ai|artificial|midjourney|dall|chatgpt|gemini|stable diffusion)"
    r"|(created|made|produced)\s+(by|with|using)\s+(ai|artificial intelligence|midjourney|dall|chatgpt|gemini)"
    r"|ai[- ]illustration|synthetic image|image generated)",
    re.I,
)


class FetchError(Exception):
    pass


@dataclass
class PageImage:
    url: str
    alt: str = ""
    caption: str = ""
    in_article: bool = False


@dataclass
class PageLink:
    url: str
    anchor: str
    kind: str          # editorial | internal | navigation | advertising
    context: str = ""


@dataclass
class Page:
    url: str
    final_url: str
    title: str = ""
    text: str = ""
    author: str = ""
    date: str = ""
    https: bool = False
    images: list[PageImage] = field(default_factory=list)
    links: list[PageLink] = field(default_factory=list)
    ad_count: int = 0
    discloses_ai_images: bool = False
    word_count: int = 0


def load_reputation() -> dict:
    with open(DATA_DIR / "source_reputation.json", encoding="utf-8") as f:
        return json.load(f)


def base_domain(host: str) -> str:
    host = (host or "").lower().split(":")[0].removeprefix("www.")
    parts = host.split(".")
    if len(parts) >= 3 and len(parts[-1]) == 2 and parts[-2] in _SLD:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def _host_is_public(host: str) -> bool:
    if os.getenv("FND_ALLOW_PRIVATE") == "1":      # tests / intranet use only
        return True
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        raise FetchError(f"Could not resolve host '{host}'.")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return False
    return True


def _download(url: str, settings: Settings, max_bytes: int, accept: str) -> tuple[bytes, str, str]:
    """GET with manual redirects so every hop is checked against the private-address rule."""
    headers = {"User-Agent": settings.user_agent, "Accept": accept}
    for _ in range(6):
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise FetchError("Only http(s) links are supported.")
        if not _host_is_public(parsed.hostname):
            raise FetchError("Links to private or local addresses are blocked.")
        try:
            r = requests.get(url, headers=headers, timeout=settings.timeout,
                             stream=True, allow_redirects=False)
        except requests.RequestException as e:
            raise FetchError(f"Request failed: {e.__class__.__name__}") from e
        if r.status_code in (301, 302, 303, 307, 308) and r.headers.get("Location"):
            url = urljoin(url, r.headers["Location"])
            continue
        if r.status_code >= 400:
            raise FetchError(f"The server answered with HTTP {r.status_code}.")
        buf = bytearray()
        for chunk in r.iter_content(65536):
            buf += chunk
            if len(buf) > max_bytes:
                break
        return bytes(buf), r.headers.get("Content-Type", ""), url
    raise FetchError("Too many redirects.")


def fetch_image(url: str, settings: Settings) -> bytes:
    data, ctype, _ = _download(url, settings, MAX_IMG, "image/*")
    if ctype and "image" not in ctype.lower() and "octet-stream" not in ctype.lower():
        raise FetchError(f"Not an image ({ctype}).")
    return data


def fetch_text(url: str, settings: Settings) -> tuple[str, str]:
    """Return (html, final_url)."""
    data, _, final = _download(url, settings, MAX_HTML, "text/html,application/xhtml+xml")
    return data.decode("utf-8", errors="replace"), final


def _soup(html: str) -> BeautifulSoup:
    try:
        return BeautifulSoup(html, "lxml")
    except Exception:
        return BeautifulSoup(html, "html.parser")


def _meta(soup: BeautifulSoup, *names: str) -> str:
    for n in names:
        tag = soup.find("meta", attrs={"property": n}) or soup.find("meta", attrs={"name": n})
        if tag and tag.get("content"):
            return tag["content"].strip()
    return ""


def _extract_text(html: str, soup: BeautifulSoup) -> str:
    try:
        import trafilatura
        out = trafilatura.extract(html, include_comments=False, include_tables=False)
        if out and len(out.split()) > 30:
            return out
    except Exception:
        pass
    root = soup.find("article") or soup.find("main") or soup.body or soup
    paras = [p.get_text(" ", strip=True) for p in root.find_all("p")]
    return "\n".join(p for p in paras if len(p.split()) > 3)


def parse_page(html: str, url: str) -> Page:
    rep = load_reputation()
    soup = _soup(html)
    host = urlparse(url).hostname or ""
    page = Page(url=url, final_url=url, https=url.startswith("https://"))

    h1 = soup.find("h1")
    page.title = (_meta(soup, "og:title", "twitter:title")
                  or (h1.get_text(" ", strip=True) if h1 else "")
                  or (soup.title.get_text(strip=True) if soup.title else ""))
    page.author = _meta(soup, "author", "article:author", "parsely-author")
    if not page.author:
        a = soup.find(attrs={"rel": "author"}) or soup.find(attrs={"itemprop": "author"}) \
            or soup.find(class_=re.compile(r"byline|author", re.I))
        page.author = a.get_text(" ", strip=True)[:80] if a else ""
    t = soup.find("time")
    page.date = (_meta(soup, "article:published_time", "datePublished", "date", "pubdate")
                 or (t.get("datetime", "") if t else ""))
    page.text = _extract_text(html, soup)
    page.word_count = len(page.text.split())

    # ad density
    ad_domains = rep["ad_domains"]
    for tag in soup.find_all(["script", "iframe"], src=True):
        if any(d in tag["src"] for d in ad_domains):
            page.ad_count += 1
    page.ad_count += len(soup.find_all(class_=re.compile(r"(^|[-_ ])ad(s|vert|vertisement)?([-_ ]|$)", re.I)))

    root = soup.find("article") or soup.find("main") or soup.body or soup
    own = base_domain(host)

    # images: og:image first, then article images
    seen: set[str] = set()
    og = _meta(soup, "og:image", "twitter:image")
    if og:
        page.images.append(PageImage(urljoin(url, og), alt="(social preview image)", in_article=True))
        seen.add(urljoin(url, og))
    for img in root.find_all("img"):
        src = img.get("src") or img.get("data-src") or ""
        if not src or src.startswith("data:"):
            continue
        full = urljoin(url, src)
        low = full.lower()
        if full in seen or any(k in low for k in ("logo", "icon", "sprite", "avatar", "pixel", "tracking", "badge", ".svg", ".gif")):
            continue
        try:
            if int(re.sub(r"\D", "", str(img.get("width", "")) or "0") or 0) in range(1, 120):
                continue
        except ValueError:
            pass
        fig = img.find_parent("figure")
        cap = fig.find("figcaption").get_text(" ", strip=True) if fig and fig.find("figcaption") else ""
        seen.add(full)
        page.images.append(PageImage(full, alt=(img.get("alt") or "").strip(), caption=cap, in_article=True))

    # links
    for a in root.find_all("a", href=True):
        href = urljoin(url, a["href"])
        p = urlparse(href)
        if p.scheme not in ("http", "https") or not p.hostname:
            continue
        anchor = a.get_text(" ", strip=True)
        hostl = p.hostname.lower()
        classes = " ".join(a.get("class", [])) + " " + " ".join(
            c for par in a.parents for c in (par.get("class") or []) if par.name in ("nav", "div", "ul", "section"))
        in_nav = a.find_parent(["nav", "header", "footer", "aside"]) is not None
        if any(d in hostl for d in ad_domains) or "sponsored" in (a.get("rel") or []):
            kind = "advertising"
        elif in_nav or re.search(r"nav|menu|footer|share|social|breadcrumb", classes, re.I):
            kind = "navigation"
        elif base_domain(hostl) == own:
            kind = "internal"
        elif a.find_parent(["p", "li", "blockquote", "figcaption"]):
            kind = "editorial"
        else:
            kind = "navigation"
        parent = a.find_parent(["p", "li", "blockquote"])
        page.links.append(PageLink(href, anchor, kind, parent.get_text(" ", strip=True)[:400] if parent else ""))

    # does the page openly label images as AI-made?
    cues = " ".join([i.alt + " " + i.caption for i in page.images])
    page.discloses_ai_images = bool(DISCLOSURE_RE.search(cues)) or bool(
        re.search(r"(image|photo|illustration|picture|graphic)[^.]{0,60}(ai[- ]generated|generated (by|with|using) ai)", page.text, re.I))
    return page


def fetch_page(url: str, settings: Settings) -> Page:
    html, final = fetch_text(url, settings)
    page = parse_page(html, final)
    page.url = url
    return page
