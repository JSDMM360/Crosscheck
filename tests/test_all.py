import io
import os
import threading
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler

import numpy as np
import pytest
from PIL import Image, PngImagePlugin

os.environ["FND_ALLOW_PRIVATE"] = "1"
os.environ["FND_DISABLE_HF"] = "1"

from fnd.config import FusionConfig, Settings          # noqa: E402
from fnd.evidence_branches import corroboration_branch  # noqa: E402
from fnd.fusion import fuse                              # noqa: E402
from fnd.image_branch import aggregate, analyze_image    # noqa: E402
from fnd.pipeline import analyze                         # noqa: E402
from fnd.schema import BranchResult                      # noqa: E402
from fnd.text_branch import text_branch                  # noqa: E402

S = Settings(use_hf_image_model=False)
rng = np.random.default_rng(0)


def _photo_like(size=(640, 480)):
    return Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))


def png_with_sd_params():
    info = PngImagePlugin.PngInfo()
    info.add_text("parameters", "a cat\nSteps: 30, Sampler: Euler a, CFG scale: 7, Seed: 1")
    b = io.BytesIO(); _photo_like((1024, 1024)).save(b, "PNG", pnginfo=info); return b.getvalue()


def png_with_iptc_marker():
    b = io.BytesIO(); _photo_like().save(b, "PNG")
    return b.getvalue() + b"http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"


def jpeg_with_c2pa_gemini():
    b = io.BytesIO(); _photo_like().save(b, "JPEG")
    return b.getvalue() + b"jumbc2pa claim_generator Google Gemini"


def jpeg_camera():
    im = _photo_like(); ex = Image.Exif()
    ex[0x010F], ex[0x0110] = "Canon", "EOS R6"
    ex.get_ifd(0x8769)[0x829A] = (1, 250)
    b = io.BytesIO(); im.save(b, "JPEG", exif=ex); return b.getvalue()


def plain_jpeg():
    b = io.BytesIO(); _photo_like().save(b, "JPEG"); return b.getvalue()


@pytest.mark.parametrize("maker", [png_with_sd_params, png_with_iptc_marker, jpeg_with_c2pa_gemini])
def test_provenance_marks_ai(maker):
    r = analyze_image(maker(), "x", S)
    assert r.verdict == "likely AI-generated" and r.p_ai >= 0.9
    assert any(s.decisive for s in r.signals)


def test_camera_jpeg_not_flagged():
    r = analyze_image(jpeg_camera(), "x", S)
    assert r.p_ai < 0.6
    assert not any(s.decisive for s in r.signals)


def test_plain_jpeg_is_inconclusive_not_authentic():
    r = analyze_image(plain_jpeg(), "x", S)
    assert r.verdict == "inconclusive"


def test_corrupt_image_reports_error():
    assert analyze_image(b"not an image", "x", S).error


def _branches(ai_strength):
    img = BranchResult("image", "Image", True, 0.4 + 0.5 * ai_strength, 0.9,
                       signals={"ai_strength": ai_strength})
    return [BranchResult("text", "Text", True, 0.6, 0.7), BranchResult("source", "Source", True, 0.5, 0.5), img]


def test_ai_images_raise_fake_probability():
    base = fuse(_branches(0.0)); boosted = fuse(_branches(1.0))
    assert boosted.p_fake > base.p_fake
    assert boosted.ai_boost > 0 and boosted.ai_weight_multiplier > 1
    assert boosted.p_fake > boosted.p_fake_without_ai_adjustment


def test_shares_sum_to_one_and_missing_branch_lowers_coverage():
    r = fuse(_branches(0.5))
    assert abs(sum(x["share"] for x in r.rows) - 1) < 1e-9
    assert r.coverage < 1


def test_disclosure_reduces_penalty():
    cfg = FusionConfig()
    flagged = analyze_image(png_with_sd_params(), "x", S)
    open_ = aggregate([flagged], False, cfg).signals["ai_strength"]
    disclosed = aggregate([flagged], True, cfg).signals["ai_strength"]
    assert 0 < disclosed < open_


def test_headline_example_text_scores_high():
    b = text_branch("cats linked to terrorists largest highest conducted in feline history", "")
    assert b.p_fake > 0.5 and b.signals["stacked_superlatives"] == 1


RSS = """<rss><channel>
<item><title>Central bank holds interest rates steady - Reuters</title><link>http://r/1</link><source url="https://www.reuters.com">Reuters</source></item>
<item><title>Central bank holds interest rates steady - Reuters Copy</title><link>http://r/2</link><source url="https://www.reuters.com">Reuters</source></item>
<item><title>Unrelated cooking story - Blog</title><link>http://b/1</link><source url="https://blog.example.org">Blog</source></item>
</channel></rss>"""


def test_corroboration_dedups_outlets():
    b = corroboration_branch("Central bank holds interest rates steady", S, rss_text=RSS)
    assert b.available and b.signals["independent_matches"] == 1 and b.p_fake < 0.5


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    d = tmp_path_factory.mktemp("site")
    (d / "ai.png").write_bytes(png_with_sd_params())
    (d / "cam.jpg").write_bytes(jpeg_camera())
    body = "<p>" + "The ministry said on Tuesday that the policy will begin in March, according to a statement. " * 8 + "</p>"
    (d / "a.html").write_text(f"""<html><head><title>Policy to begin in March</title>
      <meta name="author" content="A. Reporter"><meta property="article:published_time" content="2026-10-01"></head>
      <body><article><h1>Policy to begin in March</h1>{body}
      <figure><img src="ai.png" width="800" alt="Illustration"><figcaption>Photo</figcaption></figure>
      <img src="cam.jpg" width="800" alt="Press conference"><p>See <a href="https://www.reuters.com/world/x">Reuters report</a>.</p>
      </article></body></html>""")
    srv = HTTPServer(("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(d)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_pipeline_url_with_ai_image(site, monkeypatch):
    import fnd.evidence_branches as ev
    real_get = ev.requests.get

    def flaky(url, *a, **k):          # only the outside world is "down"
        if "127.0.0.1" in url:
            return real_get(url, *a, **k)
        raise ev.requests.ConnectionError()
    monkeypatch.setattr(ev.requests, "get", flaky)
    res = analyze(url=f"{site}/a.html", settings=S)
    names = {b.name: b for b in res.branches}
    assert names["image"].available and names["image"].signals["images_flagged_ai"] == 1
    assert names["source"].available and names["link"].available
    assert not names["corroboration"].available          # network failure handled gracefully
    assert res.fusion.ai_boost > 0 and res.fusion.p_fake > res.fusion.p_fake_without_ai_adjustment


def test_pipeline_image_only():
    res = analyze(uploaded=[("g.png", png_with_sd_params())], settings=S)
    assert [b.name for b in res.branches if b.available] == ["image"]
    assert res.fusion.p_fake > 0.6 and res.fusion.confidence <= 0.75


def test_private_urls_blocked_by_default(monkeypatch):
    monkeypatch.delenv("FND_ALLOW_PRIVATE")
    res = analyze(url="http://127.0.0.1:9/x", settings=S)
    assert any("blocked" in n for n in res.notes)
