# Crosscheck: multimodal fake news detection

Give it a headline, a news link, an image, or any mix. It runs several evidence branches, fuses them into one
fake-news probability, and shows how much each branch counted.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Recommended for image detection (large download):
pip install torch transformers
streamlit run app.py
```

Optional extras:

- Local LLM plausibility check: `ollama pull llama3.2`, run `ollama serve`, tick the Ollama box in the sidebar.
- Fact-check records: `export GOOGLE_FACTCHECK_API_KEY=...` (Google Fact Check Tools API).
- Train your own text model: `python train_text_model.py --csv data.csv --group-col source`.
- Tests: `pip install pytest && python -m pytest tests`.

## How it works

| Branch | Needs | What it looks at |
|---|---|---|
| Writing style | text | clickbait phrases, sensational words, caps, stacked superlatives, missing attribution; plus your trained TF-IDF model if present |
| Claim verifiability | text | checkable specifics (entities, numbers, attributed quotes); optional Ollama plausibility |
| Source and page metadata | link | domain reputation list, impersonation of known outlets, HTTPS, byline, date, ad density |
| Link context | link | who the article cites, how relevant the cited pages are |
| Corroboration | headline | independent outlets reporting the same story (Google News RSS), de-duplicated per outlet |
| Fact-check records | headline + API key | published fact-check ratings |
| Image authenticity | image or link | whether images were generated rather than captured (below) |

### Detecting generated images

Per image, strongest evidence first:

1. **Provenance metadata** (decisive when present): IPTC `trainedAlgorithmicMedia`, C2PA manifests naming a generator,
   generator names in EXIF/XMP/PNG text, Stable Diffusion and ComfyUI parameter chunks.
2. **Learned detector**: Hugging Face image-classification models listed in `FND_IMAGE_MODELS`
   (default `umm-maybe/AI-image-detector`; comma-separate several to average them).
3. **Weak heuristics**: typical generator resolutions, missing camera EXIF, noise uniformity.

Limits you should know about:

- Metadata is trivially stripped (screenshots, social-media re-uploads). Absence of metadata proves nothing.
- Invisible watermarks such as Google's SynthID (used on Gemini images) cannot be read without the vendor's own
  detector, so a Gemini image with stripped metadata is only caught if the learned detector recognises it.
- Public detectors trail new generators and degrade on resized or recompressed images. Treat them as one signal.
- The heuristics are uncalibrated and carry little weight on purpose.

### Fusion and the AI-image rule

```
w_i   = base_weight_i x reliability_i          (image weight x (1 + gain x AI strength) if AI images found)
share = w_i / sum(w)
z     = scale x sum(share x logit(p_i)) + boost x AI strength
P     = sigmoid(z / temperature)
```

- AI strength (0..1) comes from how confident the detector is, scaled up a little if several images are flagged.
- If the page openly labels the image as AI-made (caption or alt text), the penalty is cut to 40%, because
  publishers legitimately use AI illustrations.
- Missing branches drop out and the rest renormalise. Confidence shrinks with low coverage and with disagreement.
- All of these numbers are in `fnd/config.py` and adjustable from the sidebar.

## Honest status

- The base weights, thresholds and heuristic scores are **hand-set priors**. They have not been fitted to a
  labelled dataset, so the percentages are not calibrated probabilities. To calibrate, collect branch outputs for
  labelled stories, fit a logistic regression on them, and replace the weights and `temperature`.
- The style heuristic and the reputation lists encode their authors' judgement. Review `data/source_reputation.json`.
- An uploaded image alone can only tell you whether the image looks generated; it cannot tell you whether a story is true.
- Not included: CLIP image-text consistency, video, reverse image search, and non-English text models.
- Page fetching blocks private and local addresses by default (set `FND_ALLOW_PRIVATE=1` only for local testing).
