"""Crosscheck GUI. Run with:  streamlit run app.py"""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from fnd.config import FusionConfig, Settings
from fnd.pipeline import Analysis, analyze

st.set_page_config(page_title="Crosscheck", page_icon=None, layout="wide")

INK, PAPER, LINE = "#17212B", "#F4F6F8", "#D5DBE1"
FAKE, REAL, UNSURE, ACCENT = "#B3261E", "#1E6B52", "#A8710F", "#25506B"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:wght@500;600;700&display=swap');
html, body, [class*="css"], .stApp {{ font-family: 'IBM Plex Sans', system-ui, sans-serif; color: {INK}; }}
h1, h2, h3, .vlabel {{ font-family: 'Source Serif 4', Georgia, serif !important; letter-spacing: -0.01em; }}
.block-container {{ max-width: 1120px; padding-top: 2.2rem; }}
.verdict {{ background:#fff; border:1px solid {LINE}; border-left:8px solid var(--c); padding:1.1rem 1.4rem 1.3rem; margin:1.2rem 0 .6rem; }}
.vlabel {{ font-size:2.1rem; font-weight:700; color:var(--c); line-height:1.15; }}
.vstats {{ display:flex; gap:2.4rem; margin:.5rem 0 1rem; flex-wrap:wrap; }}
.vstats div {{ font-size:.95rem; color:#4A5866; }}
.vstats b {{ display:block; font-size:1.35rem; color:{INK}; font-weight:600; }}
.track {{ position:relative; height:14px; background:linear-gradient(90deg,#DCEBE4 0 35%,#F1E6CC 35% 65%,#F3D9D6 65% 100%); border:1px solid {LINE}; }}
.fill {{ position:absolute; top:0; bottom:0; left:0; background:var(--c); opacity:.9; }}
.tick {{ position:absolute; top:-4px; bottom:-4px; width:2px; background:{INK}; opacity:.55; }}
.before {{ position:absolute; top:-6px; bottom:-6px; width:3px; background:{INK}; }}
.scale {{ display:flex; justify-content:space-between; font-size:.8rem; color:#5B6875; margin-top:.35rem; }}
.note {{ font-size:.9rem; color:#4A5866; }}
div[data-testid="stForm"] {{ background:#fff; border:1px solid {LINE}; border-radius:2px; padding:1.2rem 1.4rem; }}
.stButton button, div[data-testid="stFormSubmitButton"] button {{ border-radius:2px; font-weight:600; }}
</style>
""", unsafe_allow_html=True)

st.title("Crosscheck")
st.markdown("Check a headline, a news link or an image. Crosscheck weighs the wording, the source, the links, "
            "the images and other outlets' coverage, then shows how much each one counted.")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Options")
    fetch_linked = st.checkbox("Fetch pages the article links to", value=False,
                               help="Slower. Compares each cited page's content with the article.")
    max_images = st.slider("Max images to inspect from a page", 1, 8, 4)
    use_hf = st.checkbox("Use learned AI-image detector", value=True,
                         help="Needs `torch` and `transformers`. Downloads a model on first use.")
    use_ollama = st.checkbox("Plausibility check with local LLM (Ollama)", value=False)
    ollama_model = st.text_input("Ollama model", "llama3.2", disabled=not use_ollama)
    with st.expander("Branch weights"):
        base = FusionConfig()
        weights = {k: st.slider(k, 0.0, 0.5, float(v), 0.01) for k, v in base.weights.items()}
        ai_boost = st.slider("AI-image boost (logit)", 0.0, 3.0, base.ai_boost_logit, 0.1)
        ai_gain = st.slider("AI-image weight gain", 0.0, 5.0, base.ai_weight_gain, 0.1)
    st.caption("Weights are hand-set starting values, not learned from data.")

# ------------------------------------------------------------------ input form
with st.form("inputs"):
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        headline = st.text_area("Headline or claim (optional)", height=90,
                                placeholder="Paste a headline or the claim you want checked")
        url = st.text_input("News link (optional)", placeholder="https://")
    with c2:
        files = st.file_uploader("Image (optional)", type=["png", "jpg", "jpeg", "webp"],
                                 accept_multiple_files=True)
    go = st.form_submit_button("Check this story", type="primary")

if go:
    if not (headline.strip() or url.strip() or files):
        st.warning("Add a headline, a link or an image first.")
    else:
        settings = Settings(max_images=max_images, fetch_linked_pages=fetch_linked, use_hf_image_model=use_hf,
                            use_ollama=use_ollama, ollama_model=ollama_model)
        cfg = FusionConfig(weights=weights, ai_boost_logit=ai_boost, ai_weight_gain=ai_gain)
        with st.spinner("Collecting evidence from each branch..."):
            st.session_state["res"] = analyze(headline, url, [(f.name, f.getvalue()) for f in files], settings, cfg)


# ------------------------------------------------------------------ rendering
def verdict_card(res: Analysis) -> None:
    f = res.fusion
    color = {"likely fake": FAKE, "likely real": REAL}.get(f.label, UNSURE)
    adj = ""
    if f.ai_boost > 0:
        adj = f'<div class="before" style="left:{f.p_fake_without_ai_adjustment*100:.1f}%"></div>'
    st.markdown(f"""
<div class="verdict" style="--c:{color}">
  <div class="vlabel">{f.label.capitalize()}</div>
  <div class="vstats">
    <div><b>{f.p_fake*100:.0f}%</b>probability the story is fake</div>
    <div><b>{f.confidence*100:.0f}%</b>confidence in the label</div>
    <div><b>{f.coverage*100:.0f}%</b>of evidence types available</div>
  </div>
  <div class="track"><div class="fill" style="width:{f.p_fake*100:.1f}%"></div>
    <div class="tick" style="left:35%"></div><div class="tick" style="left:65%"></div>{adj}</div>
  <div class="scale"><span>real</span><span>uncertain band 35% to 65%</span><span>fake</span></div>
</div>""", unsafe_allow_html=True)
    if f.ai_boost > 0:
        st.markdown(f'<div class="note">The black marker shows where the score would sit without the AI-generated '
                    f'image adjustment ({f.p_fake_without_ai_adjustment*100:.0f}%).</div>', unsafe_allow_html=True)
    for s in f.summary:
        st.markdown(f"- {s}")
    for c in f.caveats:
        st.caption(c)
    for n in res.notes:
        st.warning(n)


def stats_tab(res: Analysis) -> None:
    f = res.fusion
    by = {r["branch"]: r for r in f.rows}
    table = []
    for b in res.branches:
        r = by.get(b.name)
        table.append({
            "Branch": b.title, "Status": "used" if r else "not run",
            "P(fake)": b.p_fake if r else None, "Reliability": b.reliability if r else None,
            "Base weight": r["base_weight"] if r else None, "Effective weight": r["effective_weight"] if r else None,
            "Share of decision": r["share"] if r else None, "Push (logit)": r["contribution"] if r else None,
        })
    df = pd.DataFrame(table)
    st.dataframe(df, hide_index=True, use_container_width=True, column_config={
        "P(fake)": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
        "Reliability": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
        "Share of decision": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=1),
        "Base weight": st.column_config.NumberColumn(format="%.2f"),
        "Effective weight": st.column_config.NumberColumn(format="%.3f"),
        "Push (logit)": st.column_config.NumberColumn(format="%+.2f"),
    })
    st.caption("Positive push moves the result toward fake, negative toward real. Branches that did not run "
               "are left out and the remaining shares are rescaled.")

    bars = [{"Part": r["title"], "Push": r["contribution"]} for r in f.rows]
    if f.ai_boost > 0:
        bars.append({"Part": "AI-image adjustment", "Push": f.ai_boost})
    if bars:
        d = pd.DataFrame(bars)
        d["Direction"] = d["Push"].apply(lambda v: "toward fake" if v > 0 else "toward real")
        st.altair_chart(
            alt.Chart(d).mark_bar().encode(
                y=alt.Y("Part:N", sort="-x", title=None),
                x=alt.X("Push:Q", title="Push on the final score (logit units)"),
                color=alt.Color("Direction:N", legend=alt.Legend(title=None, orient="bottom"),
                                scale=alt.Scale(domain=["toward fake", "toward real"], range=[FAKE, REAL])),
                tooltip=["Part", alt.Tooltip("Push:Q", format="+.2f")]).properties(height=max(120, 38 * len(d))),
            use_container_width=True)

    st.subheader("AI-generated image adjustment")
    a, b, c, d_ = st.columns(4)
    a.metric("AI strength", f"{f.ai_strength:.2f}", help="0 means no generated images found, 1 means certain.")
    b.metric("Image weight multiplier", f"x{f.ai_weight_multiplier:.2f}")
    c.metric("Added push toward fake", f"{f.ai_boost:+.2f}")
    d_.metric("P(fake) before to after", f"{f.p_fake_without_ai_adjustment*100:.0f}% to {f.p_fake*100:.0f}%")
    if f.ai_strength == 0:
        st.caption("No AI-generated images were detected, so no adjustment was applied.")

    st.subheader("Confidence")
    a, b, c, d_ = st.columns(4)
    a.metric("Raw confidence", f"{f.confidence_raw*100:.0f}%")
    b.metric("Evidence coverage", f"{f.coverage*100:.0f}%")
    c.metric("Branch disagreement", f"{f.disagreement:.2f}")
    d_.metric("Final confidence", f"{f.confidence*100:.0f}%")
    with st.expander("How the score is computed"):
        st.markdown(
            "- Each branch gives P(fake) and a reliability between 0 and 1.\n"
            "- Effective weight = base weight x reliability. The image weight is multiplied by "
            "(1 + gain x AI strength) when generated images are found.\n"
            "- Share = effective weight / sum of effective weights.\n"
            "- Final logit = scale x sum(share x branch logit) + AI boost, where boost = boost x AI strength.\n"
            "- P(fake) = sigmoid(final logit / temperature).\n"
            "- Confidence shrinks when evidence coverage is low or branches disagree.")


def images_tab(res: Analysis) -> None:
    if not res.images:
        st.info("No images were uploaded or found on the page.")
        return
    for r in res.images:
        col_a, col_b = st.columns([1, 3], gap="large")
        with col_a:
            if r.thumbnail:
                st.image(r.thumbnail, use_container_width=True)
        with col_b:
            st.markdown(f"**{r.source}**")
            if r.error:
                st.error(r.error)
                continue
            color = {"likely AI-generated": FAKE, "likely authentic": REAL}.get(r.verdict, UNSURE)
            st.markdown(f'<span style="color:{color};font-weight:600">{r.verdict}</span> '
                        f'(P(AI) {r.p_ai*100:.0f}%, evidence reliability {r.reliability:.2f}, {r.width}x{r.height})',
                        unsafe_allow_html=True)
            if r.disclosed:
                st.caption("The page labels this image as AI-made.")
            st.dataframe(pd.DataFrame([{
                "Signal": s.name, "P(AI)": s.p_ai, "Weight": s.weight,
                "Decisive": s.decisive, "What it found": s.detail} for s in r.signals]),
                hide_index=True, use_container_width=True,
                column_config={"P(AI)": st.column_config.NumberColumn(format="%.2f")})
        st.divider()


def evidence_tab(res: Analysis) -> None:
    if res.page:
        st.markdown(f"**Page:** {res.page.title or res.page.final_url}  \n"
                    f"{res.page.word_count} words, {len(res.page.links)} links, {len(res.page.images)} images found")
    for b in res.branches:
        with st.expander(f"{b.title} ({'used' if b.available else 'not run'})", expanded=False):
            for n in b.notes:
                st.caption(n)
            if b.signals:
                st.json(b.signals, expanded=False)
            for e in b.evidence:
                st.markdown(f"- {e['title']}  \n  <span class='note'>{e['detail']}</span>", unsafe_allow_html=True)


if "res" in st.session_state:
    res: Analysis = st.session_state["res"]
    verdict_card(res)
    t1, t2, t3 = st.tabs(["Weightages and statistics", "Images", "Evidence and notes"])
    with t1:
        stats_tab(res)
    with t2:
        images_tab(res)
    with t3:
        evidence_tab(res)
    st.caption(f"Analysis took {res.seconds:.1f} s. Crosscheck supports human judgement and is not a final "
               "authority on truth.")
