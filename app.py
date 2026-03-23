"""
Earnings Call "Dividend Risk" Early Warning System
===================================================
ISOM5240 Group Project
Streamlit Cloud deployment file.

Pipeline Architecture:
  Pipeline 1 → Fine-tuned FinBERT  (financial sentiment)
  Pipeline 2 → Fine-tuned BERT     (dividend risk classifier, PRIMARY)
  Pipeline 3 → BART-large-mnli     (zero-shot topic check)
  Aggregator → Weighted Danger Index (0–100)
"""

import re
import time

import numpy as np
import pandas as pd
import streamlit as st
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dividend Risk Early Warning System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS  ← replace model IDs after pushing to HuggingFace Hub
# ─────────────────────────────────────────────────────────────────────────────
SENTIMENT_MODEL_ID = "ruirui0506/finbert-dividend-sentiment"   # Pipeline 1
RISK_MODEL_ID      = "ruirui0506/dividend-risk-bert"           # Pipeline 2
ZEROSHOT_MODEL_ID  = "facebook/bart-large-mnli"                   # Pipeline 3

DANGER_LABELS = [
    "dividend cut risk",
    "liquidity stress",
    "debt covenant breach",
    "capital preservation",
    "payout suspension",
]

# Default pipeline weights (user can adjust in sidebar)
WEIGHTS = {"sentiment": 0.25, "risk": 0.50, "topic": 0.25}

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING  (cached so Streamlit doesn't reload on every interaction)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_sentiment_pipeline():
    """Pipeline 1 – fine-tuned FinBERT for financial sentiment."""
    try:
        tok = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_ID)
        mdl = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_ID)
        return pipeline(
            "text-classification", model=mdl, tokenizer=tok,
            device=-1, top_k=None,
        )
    except Exception:
        # Fallback: base FinBERT (before fine-tuning is pushed to Hub)
        st.warning("⚠️ Fine-tuned sentiment model not found — using base ProsusAI/finbert.")
        return pipeline(
            "text-classification", model="ProsusAI/finbert",
            device=-1, top_k=None,
        )


@st.cache_resource(show_spinner=False)
def load_risk_pipeline():
    """Pipeline 2 – fine-tuned BERT for dividend risk classification (PRIMARY)."""
    try:
        tok = AutoTokenizer.from_pretrained(RISK_MODEL_ID)
        mdl = AutoModelForSequenceClassification.from_pretrained(RISK_MODEL_ID)
        return pipeline(
            "text-classification", model=mdl, tokenizer=tok,
            device=-1, top_k=None,
        )
    except Exception:
        # Fallback: base DistilBERT (before fine-tuning is pushed)
        st.warning("⚠️ Fine-tuned risk model not found — using base distilbert-base-uncased.")
        return pipeline(
            "text-classification", model="distilbert-base-uncased",
            device=-1, top_k=None,
        )


@st.cache_resource(show_spinner=False)
def load_zeroshot_pipeline():
    """Pipeline 3 – BART-large-mnli for zero-shot dividend topic detection."""
    return pipeline(
        "zero-shot-classification",
        model=ZEROSHOT_MODEL_ID, device=-1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TEXT UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    """Split transcript into overlapping word-level windows for batch inference."""
    words = text.split()
    chunks, step = [], chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if len(chunk.strip()) > 20:
            chunks.append(chunk)
    return chunks


def get_sentences(text: str) -> list[str]:
    """Split text into sentences; drop very short ones."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if len(s.split()) >= 6]


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE RUNNERS
# ─────────────────────────────────────────────────────────────────────────────
def run_sentiment_analysis(chunks: list[str], sent_pipe) -> dict:
    """
    Pipeline 1 — FinBERT financial sentiment.
    Aggregates per-chunk scores; maps high negativity → high danger contribution.
    """
    pos_scores, neg_scores, neu_scores = [], [], []

    for chunk in chunks:
        result = sent_pipe(chunk[:512], truncation=True)
        scores = {item["label"].lower(): item["score"] for item in result[0]}
        pos_scores.append(scores.get("positive", 0.0))
        neg_scores.append(scores.get("negative", 0.0))
        neu_scores.append(scores.get("neutral",  0.0))

    avg_pos = float(np.mean(pos_scores))
    avg_neg = float(np.mean(neg_scores))
    avg_neu = float(np.mean(neu_scores))

    return {
        "positive":    avg_pos,
        "negative":    avg_neg,
        "neutral":     avg_neu,
        "danger_score": avg_neg * 100,   # 0–100: more negative → higher danger
    }


def run_risk_classification(chunks: list[str], risk_pipe) -> dict:
    """
    Pipeline 2 — Fine-tuned dividend risk classifier (PRIMARY model).
    LABEL_0 = Low Risk, LABEL_1 = Medium Risk, LABEL_2 = High Risk.
    Aggregates probabilities across all chunks via averaging.
    """
    low_probs, med_probs, high_probs = [], [], []

    for chunk in chunks:
        result = risk_pipe(chunk[:512], truncation=True)
        scores = {item["label"]: item["score"] for item in result[0]}
        # Accept both LABEL_X naming and human-readable naming
        low_probs.append(scores.get("LABEL_0", scores.get("Low Risk",    0.333)))
        med_probs.append(scores.get("LABEL_1", scores.get("Medium Risk", 0.333)))
        high_probs.append(scores.get("LABEL_2", scores.get("High Risk",  0.333)))

    avg_low  = float(np.mean(low_probs))
    avg_med  = float(np.mean(med_probs))
    avg_high = float(np.mean(high_probs))

    # High risk contributes full weight; medium risk half weight
    danger_score = min((avg_high * 100) + (avg_med * 50), 100.0)

    return {
        "low_prob":     avg_low,
        "med_prob":     avg_med,
        "high_prob":    avg_high,
        "danger_score": danger_score,
        "dominant_class": ["Low Risk", "Medium Risk", "High Risk"][
            int(np.argmax([avg_low, avg_med, avg_high]))
        ],
    }


def run_zeroshot_analysis(text: str, zs_pipe) -> dict:
    """
    Pipeline 3 — BART zero-shot classification.
    Checks how strongly the transcript covers dividend-risk topics.
    Uses the first 1 000 words to keep runtime acceptable on CPU.
    """
    sample_text = " ".join(text.split()[:1000])
    result = zs_pipe(sample_text, DANGER_LABELS, multi_label=True)
    label_scores = dict(zip(result["labels"], result["scores"]))
    avg_topic_score = float(np.mean(list(label_scores.values()))) * 100

    return {
        "label_scores": label_scores,
        "danger_score": avg_topic_score,
        "top_topic":    result["labels"][0],
    }


# ─────────────────────────────────────────────────────────────────────────────
# DANGER INDEX
# ─────────────────────────────────────────────────────────────────────────────
def compute_danger_index(s_score: float, r_score: float, t_score: float) -> float:
    """Weighted fusion of three pipeline danger scores → single 0–100 index."""
    raw = (
        WEIGHTS["sentiment"] * s_score
        + WEIGHTS["risk"]    * r_score
        + WEIGHTS["topic"]   * t_score
    )
    return round(min(max(raw, 0.0), 100.0), 1)


def get_signal(index: float) -> tuple[str, str]:
    """Map index to (signal_label, hex_color)."""
    if index <= 30:
        return "HEALTHY",  "#1D9E75"
    elif index <= 55:
        return "CAUTION",  "#BA7517"
    elif index <= 75:
        return "DANGER",   "#D85A30"
    else:
        return "CRITICAL", "#A32D2D"


# ─────────────────────────────────────────────────────────────────────────────
# SENTENCE-LEVEL HIGHLIGHTING
# ─────────────────────────────────────────────────────────────────────────────
def highlight_risky_sentences(
    text: str, sent_pipe, risk_pipe, max_sentences: int = 60
) -> list[dict]:
    """Score each sentence; return top-5 highest-risk excerpts."""
    sentences = get_sentences(text)[:max_sentences]
    scored = []

    for sent in sentences:
        if len(sent.split()) < 5:
            continue
        try:
            s_res  = sent_pipe(sent[:512], truncation=True)
            s_dict = {item["label"].lower(): item["score"] for item in s_res[0]}
            neg    = s_dict.get("negative", 0.0)

            r_res  = risk_pipe(sent[:512], truncation=True)
            r_dict = {item["label"]: item["score"] for item in r_res[0]}
            high   = r_dict.get("LABEL_2", r_dict.get("High Risk", 0.0))

            scored.append({"sentence": sent, "score": neg * 0.4 + high * 0.6})
        except Exception:
            continue

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:5]


# ─────────────────────────────────────────────────────────────────────────────
# GAUGE CHART  (pure HTML/SVG — no external chart library required)
# ─────────────────────────────────────────────────────────────────────────────
def make_gauge(index: float, signal: str, color: str):
    """Renders a semicircle gauge using pure HTML/SVG via st.markdown."""

    # Gauge arc: 180 degrees semicircle
    # Convert index (0-100) to angle (180 to 0 degrees, left to right)
    angle_deg  = 180 - (index / 100 * 180)
    angle_rad  = angle_deg * 3.14159 / 180
    needle_x   = 150 + 110 * (0 if angle_deg == 90 else
                  __import__('math').cos(angle_rad))
    needle_y   = 160 - 110 * __import__('math').sin(angle_rad)

    # Zone colours
    zone_html = (
        # Green zone  0-30   (arc from 180° to 126°)
        "<path d='M 40 160 A 110 110 0 0 1 95 65' "
        "fill='none' stroke='#1D9E75' stroke-width='18' stroke-linecap='butt'/>"
        # Yellow zone 30-55  (arc from 126° to 90°)
        "<path d='M 95 65 A 110 110 0 0 1 150 50' "
        "fill='none' stroke='#BA7517' stroke-width='18' stroke-linecap='butt'/>"
        # Orange zone 55-75  (arc from 90° to 54°)
        "<path d='M 150 50 A 110 110 0 0 1 205 65' "
        "fill='none' stroke='#D85A30' stroke-width='18' stroke-linecap='butt'/>"
        # Red zone    75-100 (arc from 54° to 0°)
        "<path d='M 205 65 A 110 110 0 0 1 260 160' "
        "fill='none' stroke='#A32D2D' stroke-width='18' stroke-linecap='butt'/>"
    )

    gauge_html = f"""
    <div style='text-align:center;padding:10px 0 0 0'>
      <svg viewBox='0 0 300 190' width='100%' style='max-width:320px'>
        <!-- Background arc -->
        <path d='M 40 160 A 110 110 0 0 1 260 160'
              fill='none' stroke='#e8e8e8' stroke-width='20'/>
        <!-- Coloured zone arcs -->
        {zone_html}
        <!-- Needle -->
        <line x1='150' y1='160'
              x2='{needle_x:.1f}' y2='{needle_y:.1f}'
              stroke='{color}' stroke-width='3'
              stroke-linecap='round'/>
        <!-- Centre dot -->
        <circle cx='150' cy='160' r='6' fill='{color}'/>
        <!-- Zone labels -->
        <text x='28'  y='178' font-size='9' fill='#1D9E75' text-anchor='middle'>0</text>
        <text x='88'  y='62'  font-size='9' fill='#BA7517' text-anchor='middle'>30</text>
        <text x='150' y='36'  font-size='9' fill='#D85A30' text-anchor='middle'>55</text>
        <text x='212' y='62'  font-size='9' fill='#A32D2D' text-anchor='middle'>75</text>
        <text x='272' y='178' font-size='9' fill='#A32D2D' text-anchor='middle'>100</text>
        <!-- Index value -->
        <text x='150' y='148' font-size='28' font-weight='700'
              fill='{color}' text-anchor='middle'>{index}</text>
        <!-- Label -->
        <text x='150' y='180' font-size='11' fill='gray' text-anchor='middle'>
          Dividend Danger Index
        </text>
      </svg>
      <div style='font-size:1.1em;font-weight:700;color:{color};
                  margin-top:2px;letter-spacing:1px'>{signal}</div>
    </div>
    """
    st.markdown(gauge_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SAMPLE TRANSCRIPTS
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_SAFE = """
Good morning, and thank you for joining our quarterly earnings call.
I am pleased to report that this quarter has been exceptional for our company.
Revenue grew 12% year-over-year, driven by strong demand across all business segments.
Free cash flow reached a record high of 2.8 billion dollars, providing ample coverage
for our dividend commitments.

Our payout ratio remains conservative at 42%, well within our target range of 40 to 55%.
The board of directors has approved a 5% increase in the quarterly dividend, reflecting
our confidence in the sustainability of our cash generation and the strength of our
balance sheet.

We have reduced our net debt by 800 million dollars this quarter, and our interest
coverage ratio stands at a healthy 8.2 times. Our liquidity position is robust, with
3.4 billion dollars in available credit facilities and no material debt maturities
until 2027.

Looking ahead, we are raising our full-year guidance and expect continued strong
performance driven by our diversified revenue streams. Capital allocation remains
a top priority, and returning value to shareholders through dividends and share
repurchases is central to our strategy. We are committed to growing the dividend
consistently over the long term.
"""

SAMPLE_RISKY = """
Thank you for joining today's call. I want to be transparent about the challenges
we are currently navigating. This quarter presented significant headwinds across our
core business segments, with revenue declining 18% year-over-year due to pricing
pressure and volume loss in our key markets.

Free cash flow turned negative for the second consecutive quarter, coming in at
minus 340 million dollars. We have drawn down substantially on our revolving credit
facility, and our available liquidity has decreased to 680 million dollars.
Our net leverage ratio has increased to 5.8 times EBITDA, approaching the
covenant threshold of 6.0 times.

Regarding our dividend, the board is actively reviewing the sustainability of the
current payout level. We are prioritizing debt reduction and preserving capital to
navigate this difficult period. We are exploring all options to strengthen our
balance sheet, including potential asset divestitures and a review of our capital
return program.

Our payout ratio has risen to 127% of free cash flow, which is clearly unsustainable.
Management is committed to taking decisive action to restore financial stability.
We acknowledge that some of these measures may be difficult in the short term but
are necessary to ensure the long-term health of the company. We will provide further
clarity on our capital allocation strategy at a special investor day next month.
"""

# ─────────────────────────────────────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.title("📊 Earnings Call Dividend Risk Early Warning System")
    st.caption(
        "Analyzes earnings call transcripts using **3 deep learning pipelines** "
        "to compute a real-time **Dividend Danger Index (0–100)**."
    )

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("📋 Company Info")
        company_name = st.text_input("Company Name", placeholder="e.g. Realty Income Corp")
        ticker       = st.text_input("Stock Ticker",  placeholder="e.g. O")
        quarter      = st.selectbox(
            "Earnings Quarter",
            ["Q1 2024","Q2 2024","Q3 2024","Q4 2024",
             "Q1 2025","Q2 2025","Q3 2025","Q4 2025"],
        )
        st.divider()

        st.subheader("⚖️ Pipeline Weights")
        w_s = st.slider("Sentiment  (Pipeline 1)", 0.0, 1.0, 0.25, 0.05)
        w_r = st.slider("Risk Class (Pipeline 2)", 0.0, 1.0, 0.50, 0.05)
        w_t = st.slider("Topic Check (Pipeline 3)", 0.0, 1.0, 0.25, 0.05)
        total = w_s + w_r + w_t
        if total > 0:
            w_s, w_r, w_t = w_s / total, w_r / total, w_t / total
        WEIGHTS.update({"sentiment": w_s, "risk": w_r, "topic": w_t})
        st.caption(f"Normalised → Sent: {w_s:.2f} | Risk: {w_r:.2f} | Topic: {w_t:.2f}")
        st.divider()
        st.info("Models are cached after the first load (~30 s on CPU).")

    # ── Transcript Input ──────────────────────────────────────────────────────
    st.subheader("📄 Earnings Call Transcript")
    col_left, col_right = st.columns([3, 1])

    with col_left:
        transcript = st.text_area(
            "transcript_input",
            height=260,
            placeholder=(
                "Paste the full earnings call transcript here...\n\n"
                "Tip: Include the CFO/CEO prepared remarks section for best results."
            ),
            label_visibility="collapsed",
        )

    with col_right:
        uploaded = st.file_uploader("Or upload .txt", type=["txt"])
        if uploaded:
            transcript = uploaded.read().decode("utf-8")
            st.success(f"Loaded: {uploaded.name}")

        st.markdown("**Quick test:**")
        if st.button("✅ Load safe example"):
            transcript = SAMPLE_SAFE
        if st.button("🚨 Load risky example"):
            transcript = SAMPLE_RISKY

    # ── Analyze Button ────────────────────────────────────────────────────────
    analyze = st.button(
        "🔍 Analyze Dividend Risk",
        type="primary",
        disabled=len(transcript.strip()) < 50,
    )

    if not transcript.strip():
        st.info("Paste an earnings call transcript above and click **Analyze Dividend Risk**.")
        return

    if not analyze:
        return

    # ── Load Models ───────────────────────────────────────────────────────────
    with st.spinner("Loading models — first run may take ~30 s …"):
        sent_pipe = load_sentiment_pipeline()
        risk_pipe = load_risk_pipeline()
        zs_pipe   = load_zeroshot_pipeline()

    # ── Run All Three Pipelines ───────────────────────────────────────────────
    chunks   = chunk_text(transcript)
    n_chunks = len(chunks)
    prog     = st.progress(0, text="Pipeline 1/3 — Financial Sentiment …")
    t0       = time.time()

    sentiment_res = run_sentiment_analysis(chunks, sent_pipe)
    prog.progress(33, text="Pipeline 2/3 — Dividend Risk Classifier …")

    risk_res = run_risk_classification(chunks, risk_pipe)
    prog.progress(66, text="Pipeline 3/3 — Zero-Shot Topic Check …")

    topic_res = run_zeroshot_analysis(transcript, zs_pipe)
    prog.progress(90, text="Computing Danger Index …")

    danger_index = compute_danger_index(
        sentiment_res["danger_score"],
        risk_res["danger_score"],
        topic_res["danger_score"],
    )
    signal, color = get_signal(danger_index)
    runtime = round(time.time() - t0, 1)

    prog.progress(100, text=f"Analysis complete! ({runtime} s)")
    time.sleep(0.4)
    prog.empty()

    # ── Header Row ────────────────────────────────────────────────────────────
    st.divider()
    h_col, b_col = st.columns([3, 1])
    with h_col:
        label = f"{company_name} ({ticker})" if company_name else "Selected Company"
        st.subheader(f"📈 Results — {label} · {quarter}")
        st.caption(f"Analysed **{n_chunks}** transcript chunks · Runtime: **{runtime} s**")
    with b_col:
        badge_style = {
            "HEALTHY":  ("background:#E1F5EE;color:#085041", "✅"),
            "CAUTION":  ("background:#FAEEDA;color:#633806", "⚠️"),
            "DANGER":   ("background:#FAECE7;color:#712B13", "🔶"),
            "CRITICAL": ("background:#FCEBEB;color:#791F1F", "🚨"),
        }
        bstyle, icon = badge_style[signal]
        st.markdown(
            f"<div style='{bstyle};padding:12px 16px;border-radius:8px;"
            f"text-align:center;font-weight:700;font-size:1.2em;margin-top:6px'>"
            f"{icon} {signal}</div>",
            unsafe_allow_html=True,
        )

    # ── Gauge + Breakdown ─────────────────────────────────────────────────────
    g_col, br_col = st.columns([1, 1])

    with g_col:
        make_gauge(danger_index, signal, color)

    with br_col:
        st.markdown("### Pipeline Score Breakdown")
        pipeline_rows = [
            (
                "Pipeline 1 — Financial Sentiment",
                sentiment_res["danger_score"],
                f"Negative: {sentiment_res['negative']:.1%}  |  "
                f"Positive: {sentiment_res['positive']:.1%}  |  "
                f"Neutral: {sentiment_res['neutral']:.1%}",
            ),
            (
                "Pipeline 2 — Dividend Risk Classifier  ★",
                risk_res["danger_score"],
                f"High risk: {risk_res['high_prob']:.1%}  |  "
                f"Medium: {risk_res['med_prob']:.1%}  |  "
                f"Low: {risk_res['low_prob']:.1%}  → dominant: {risk_res['dominant_class']}",
            ),
            (
                "Pipeline 3 — Zero-Shot Topic Check",
                topic_res["danger_score"],
                f"Top topic: {topic_res['top_topic']}",
            ),
        ]
        for name, score, detail in pipeline_rows:
            bar_c = (
                "#1D9E75" if score <= 30 else
                "#BA7517" if score <= 55 else
                "#D85A30" if score <= 75 else "#A32D2D"
            )
            st.markdown(f"**{name}**")
            st.markdown(
                f"<small style='color:gray'>{detail}</small>",
                unsafe_allow_html=True,
            )
            st.progress(int(score) / 100)
            st.markdown(
                f"<p style='text-align:right;color:{bar_c};font-weight:600;"
                f"margin:-10px 0 10px'>Score: {score:.1f} / 100</p>",
                unsafe_allow_html=True,
            )

    # ── Zero-Shot Topic Table & Bar ───────────────────────────────────────────
    st.divider()
    st.subheader("🏷️ Dividend-Risk Topic Relevance (Pipeline 3)")

    t_col1, t_col2 = st.columns([2, 3])
    sorted_topics = sorted(
        topic_res["label_scores"].items(), key=lambda x: x[1], reverse=True
    )
    with t_col1:
        st.dataframe(
            pd.DataFrame(sorted_topics, columns=["Risk Topic", "Score"]).assign(
                Score=lambda df: df["Score"].map("{:.1%}".format)
            ),
            hide_index=True,
            use_container_width=True,
        )
    with t_col2:
        vals   = [v for _, v in sorted_topics]
        labels = [k for k, _ in sorted_topics]
        st.bar_chart(
            pd.DataFrame({"Risk Score": vals}, index=labels),
            use_container_width=True,
            height=220,
        )

    # ── Risky Sentence Highlights ─────────────────────────────────────────────
    st.divider()
    st.subheader("⚠️ Highest-Risk Sentences Detected")
    st.caption(
        "Ranked by combined negativity (Pipeline 1) × high-risk probability (Pipeline 2)."
    )
    with st.spinner("Scoring sentences …"):
        risky_sents = highlight_risky_sentences(transcript, sent_pipe, risk_pipe)

    if risky_sents:
        for i, item in enumerate(risky_sents, 1):
            rp = item["score"] * 100
            bc = "#A32D2D" if rp > 60 else "#D85A30" if rp > 40 else "#BA7517"
            st.markdown(
                f"<div style='border-left:4px solid {bc};padding:8px 14px;"
                f"margin:5px 0;background:rgba(0,0,0,0.02);border-radius:0 6px 6px 0'>"
                f"<span style='color:{bc};font-weight:600;font-size:0.8em'>"
                f"#{i} · Risk score: {rp:.0f}/100</span><br>"
                f"<span style='font-size:0.94em'>{item['sentence']}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("No strongly risky sentences found.")

    # ── Recommendation Box ────────────────────────────────────────────────────
    st.divider()
    st.subheader("💡 System Recommendation")
    recommendations = {
        "HEALTHY": (
            "The transcript shows **healthy dividend language**. Sentiment is predominantly "
            "positive, the risk classifier indicates low cut probability, and dividend-risk "
            "topics are not prominently featured. **Action:** Monitor normally — "
            "no immediate concern."
        ),
        "CAUTION": (
            "The transcript shows **some cautionary signals**. Moderate negative sentiment "
            "or ambiguous language around cash flow and payout ratios is detected. "
            "**Action:** Review payout ratio, free cash flow coverage, and debt trajectory. "
            "Monitor closely next quarter."
        ),
        "DANGER": (
            f"The transcript contains **significant dividend risk signals**. High negativity "
            f"and elevated risk classification scores detected across multiple chunks. "
            f"Top risk topic flagged: *{topic_res['top_topic']}*. "
            f"**Action:** Reduce exposure or hedge. Review balance sheet and "
            f"dividend coverage ratio immediately."
        ),
        "CRITICAL": (
            f"The transcript triggers **critical dividend risk alerts** across all pipelines. "
            f"Strong signals of liquidity stress, negative cash flow language, or explicit "
            f"dividend review discussion detected. Top topic: *{topic_res['top_topic']}*. "
            f"**Action:** High probability of dividend cut or suspension. "
            f"Consider immediate position review."
        ),
    }
    rec_bg = {
        "HEALTHY": "#E1F5EE", "CAUTION": "#FAEEDA",
        "DANGER":  "#FAECE7", "CRITICAL": "#FCEBEB",
    }
    st.markdown(
        f"<div style='background:{rec_bg[signal]};padding:16px 20px;"
        f"border-radius:8px;border-left:5px solid {color}'>"
        f"{recommendations[signal]}</div>",
        unsafe_allow_html=True,
    )

    # ── Export ────────────────────────────────────────────────────────────────
    st.divider()
    export_df = pd.DataFrame(
        [
            {
                "Company":             company_name,
                "Ticker":              ticker,
                "Quarter":             quarter,
                "Danger Index":        danger_index,
                "Signal":              signal,
                "Sentiment Score":     round(sentiment_res["danger_score"], 2),
                "Risk Score":          round(risk_res["danger_score"], 2),
                "Topic Score":         round(topic_res["danger_score"], 2),
                "Positive Sentiment":  round(sentiment_res["positive"], 4),
                "Negative Sentiment":  round(sentiment_res["negative"], 4),
                "Low Risk Prob":       round(risk_res["low_prob"], 4),
                "Medium Risk Prob":    round(risk_res["med_prob"], 4),
                "High Risk Prob":      round(risk_res["high_prob"], 4),
                "Top Risk Topic":      topic_res["top_topic"],
                "Chunks Analysed":     n_chunks,
                "Runtime (s)":         runtime,
            }
        ]
    )
    st.download_button(
        label="📥 Export Results (CSV)",
        data=export_df.to_csv(index=False),
        file_name=f"dividend_risk_{ticker or 'result'}_{quarter.replace(' ', '_')}.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
