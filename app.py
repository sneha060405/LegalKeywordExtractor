import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
import PyPDF2
import re
import pandas as pd
import math
nltk.download('punkt')
nltk.download('stopwords')

# ─────────────────────────────────────────────
# NLTK SETUP
# ─────────────────────────────────────────────
for pkg in ["punkt", "stopwords", "averaged_perceptron_tagger", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}" if "punkt" in pkg else f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Legal Document Analyzer",
    layout="wide",
    page_icon="⚖️"
)

# ─────────────────────────────────────────────
# CSS STYLING
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stApp { background-color: #0f1117; color: #e0e0e0; }
    h1, h2, h3 { color: #c9a84c; }
    .highlight-box {
        background-color: #1a1d26;
        border-left: 4px solid #c9a84c;
        padding: 14px 18px;
        margin: 10px 0;
        border-radius: 6px;
        font-size: 15px;
        line-height: 1.6;
        color: #ddd;
    }
    .tag {
        display: inline-block;
        background: #c9a84c22;
        color: #c9a84c;
        border: 1px solid #c9a84c55;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 13px;
        margin: 3px;
    }
    .section-header {
        font-size: 20px;
        font-weight: bold;
        color: #c9a84c;
        border-bottom: 1px solid #333;
        padding-bottom: 6px;
        margin-bottom: 12px;
    }
    .risk-high { color: #ff6b6b; font-weight: bold; }
    .risk-medium { color: #ffa94d; }
    .risk-low { color: #69db7c; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
LEGAL_ACTION_WORDS = {
    "shall", "must", "agree", "liable", "responsible", "terminate", "breach",
    "pay", "penalty", "confidential", "indemnify", "obligation", "warrant",
    "represent", "covenant", "enforce", "damages", "remedy", "default",
    "assign", "notify", "disclose", "restrict", "prohibit", "authorize"
}

BOILERPLATE_PHRASES = [
    "this agreement is made", "witnesseth", "in witness whereof",
    "hereby agreed", "now therefore", "as of the date", "referred to as",
    "hereinafter referred to", "entered into as of", "made and entered"
]

RISK_KEYWORDS = {
    "high": ["terminate immediately", "liquidated damages", "indemnify", "unlimited liability",
             "irrevocable", "waive all claims", "penalty", "forfeit", "binding arbitration"],
    "medium": ["may terminate", "reasonable notice", "sole discretion", "confidential",
               "non-compete", "non-disclosure", "restrict", "prohibit"],
    "low": ["notice period", "mutual agreement", "either party", "amendment", "renewal"]
}

CLAUSE_CATEGORIES = {
    "Termination": ["terminat", "end of agreement", "cancel", "expire", "cessation"],
    "Payment": ["pay", "compensation", "fee", "invoice", "salary", "remuneration", "amount due"],
    "Liability": ["liable", "liability", "indemnif", "damages", "loss", "claim"],
    "Confidentiality": ["confidential", "non-disclosure", "nda", "proprietary", "trade secret"],
    "Obligations": ["shall", "must", "obligat", "duty", "required to", "responsible for"],
    "Dispute Resolution": ["arbitration", "mediation", "dispute", "governing law", "jurisdiction"]
}

COMMON_LEGAL_WORDS = {"shall", "may", "agreement", "hereby", "thereof", "party", "parties", "herein"}


# ─────────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────────
def clean_text(raw_text: str) -> str:
    text = re.sub(r'\n+', ' ', raw_text)
    text = re.sub(r'(\b\w)\s+(\w\b)', r'\1\2', text)  # fix broken words
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'_,?\d*', '', text)                 # remove PDF artifacts
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)         # remove non-ASCII
    text = re.sub(r'\b(\w)\.\s+(\w)\.\s+', '', text)   # remove initials noise
    return text.strip()


# ─────────────────────────────────────────────
# PDF EXTRACTOR
# ─────────────────────────────────────────────
def extract_pdf_text(uploaded_file) -> str:
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    raw = ""
    for page in pdf_reader.pages:
        raw += (page.extract_text() or "") + " "
    return clean_text(raw)


# ─────────────────────────────────────────────
# TF-IDF KEYWORDS
# ─────────────────────────────────────────────
def extract_keywords(text: str, n: int = 20):
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),      # captures bigrams like "breach of contract"
        min_df=1,
        max_features=500
    )
    X = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    scores = X.toarray()[0]
    word_scores = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)

    # Filter out pure stopword-derived pairs and very short terms
    filtered = [(w, s) for w, s in word_scores if len(w) > 3 and s > 0]
    return filtered[:n]


# ─────────────────────────────────────────────
# SENTENCE FILTER — remove noise, headings, boilerplate
# ─────────────────────────────────────────────
def is_valid_sentence(sentence: str) -> bool:
    s = sentence.strip()
    clean = s.lower()

    # Too short
    if len(s) < 40:
        return False

    # All-uppercase heading
    alpha_words = re.findall(r'[A-Za-z]+', s)
    if alpha_words and sum(1 for w in alpha_words if w.isupper()) / len(alpha_words) > 0.6:
        return False

    # Numbered list items / section headers
    if re.match(r'^[\d]+[\.\)]\s', s):
        return False

    # Boilerplate
    if any(phrase in clean for phrase in BOILERPLATE_PHRASES):
        return False

    # Likely a pure definition line ("X means Y as defined in...")
    if clean.count('"') >= 4 and len(s) < 120:
        return False

    return True


# ─────────────────────────────────────────────
# SENTENCE SCORER
# ─────────────────────────────────────────────
def score_sentences(text: str, top_keywords):
    sentences = sent_tokenize(text)
    kw_set = {w for w, _ in top_keywords}
    kw_scores = {w: s for w, s in top_keywords}

    scored = []
    for idx, sentence in enumerate(sentences):
        if not is_valid_sentence(sentence):
            continue

        words_lower = sentence.lower().split()
        word_set = set(words_lower)

        # Keyword score: bigrams + unigrams
        kw_score = 0.0
        for kw in kw_set:
            if kw in sentence.lower():
                weight = kw_scores.get(kw, 0)
                boost = 0.5 if kw in COMMON_LEGAL_WORDS else 1.0
                kw_score += weight * boost * (1.5 if ' ' in kw else 1.0)  # bigram bonus

        # Legal action word boost
        action_count = sum(1 for w in LEGAL_ACTION_WORDS if w in word_set)
        action_boost = 1 + (action_count * 0.3)

        # Position weight: early and late sentences are more important
        n = len(sentences)
        pos_norm = idx / max(n - 1, 1)
        position_weight = 1.2 if pos_norm < 0.2 else (1.1 if pos_norm > 0.8 else 1.0)

        # Penalise excessive length, but gently
        length = len(words_lower)
        length_penalty = math.sqrt(max(length, 1))

        # Repetition (ratio of unique words)
        unique_ratio = len(set(words_lower)) / max(length, 1)
        repetition_factor = 0.5 + unique_ratio  # 0.5–1.5

        final_score = (kw_score * action_boost * position_weight * repetition_factor) / length_penalty

        scored.append((sentence, final_score))

    return sorted(scored, key=lambda x: x[1], reverse=True)


# ─────────────────────────────────────────────
# DIVERSITY FILTER — Jaccard-based dedup
# ─────────────────────────────────────────────
def diverse_summary(scored_sentences, n=5, overlap_threshold=0.35):
    selected = []
    for sentence, score in scored_sentences:
        words_new = set(sentence.lower().split())
        is_duplicate = False
        for existing in selected:
            words_existing = set(existing.lower().split())
            intersection = words_new & words_existing
            union = words_new | words_existing
            jaccard = len(intersection) / max(len(union), 1)
            if jaccard > overlap_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            selected.append(sentence)
        if len(selected) == n:
            break
    return selected


# ─────────────────────────────────────────────
# CLAUSE DETECTOR
# ─────────────────────────────────────────────
def detect_clauses(text: str):
    found = {}
    text_lower = text.lower()
    for category, keywords in CLAUSE_CATEGORIES.items():
        sentences = sent_tokenize(text)
        matches = [s for s in sentences if any(kw in s.lower() for kw in keywords)]
        if matches:
            found[category] = matches[0]  # best matching sentence
    return found


# ─────────────────────────────────────────────
# RISK DETECTOR
# ─────────────────────────────────────────────
def detect_risks(text: str):
    risks = {"high": [], "medium": [], "low": []}
    text_lower = text.lower()
    for level, phrases in RISK_KEYWORDS.items():
        for phrase in phrases:
            if phrase in text_lower:
                risks[level].append(phrase)
    return risks


# ─────────────────────────────────────────────
# WORD CLOUD
# ─────────────────────────────────────────────
def render_wordcloud(text: str):
    wc = WordCloud(
        width=700, height=320,
        background_color='#1a1d26',
        colormap='YlOrBr',
        max_words=80,
        collocations=False
    ).generate(text)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    fig.patch.set_facecolor('#1a1d26')
    ax.set_facecolor('#1a1d26')
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    return fig


# ─────────────────────────────────────────────
# FREQUENCY BAR CHART
# ─────────────────────────────────────────────
def render_freq_chart(text: str, top_keywords):
    stop_words = set(stopwords.words('english'))
    words = [w for w in word_tokenize(text.lower()) if w.isalnum() and w not in stop_words]
    word_counts = Counter(words)

    labels = [kw.upper() for kw, _ in top_keywords[:12]]
    values = [word_counts.get(kw.split()[0], 0) for kw, _ in top_keywords[:12]]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor('#1a1d26')
    ax.set_facecolor('#1a1d26')
    colors = ['#c9a84c' if v == max(values) else '#8a6f30' for v in values]
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1], edgecolor='none', height=0.6)
    ax.tick_params(colors='#aaa', labelsize=9)
    ax.spines[:].set_visible(False)
    ax.xaxis.label.set_color('#aaa')
    ax.set_xlabel("Frequency", color='#aaa', fontsize=9)
    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                str(val), va='center', color='#ccc', fontsize=8)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────
st.markdown("# ⚖️ Legal Document Analyzer")
st.markdown("Extract **keywords**, generate **smart summaries**, detect **clauses & risks** — instantly.")

st.markdown("---")

st.subheader("📄 Input Document")
option = st.radio("Choose input type:", ("Paste Text", "Upload PDF"), horizontal=True)

text = ""

if option == "Paste Text":
    text = st.text_area("Paste your legal document here:", height=260,
                        placeholder="Paste contract, NDA, lease, or any legal text...")

elif option == "Upload PDF":
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file:
        with st.spinner("Extracting PDF text..."):
            text = extract_pdf_text(uploaded_file)
        st.success(f"✅ PDF extracted — {len(text.split())} words found")
        with st.expander("Preview extracted text"):
            st.write(text[:1500] + ("..." if len(text) > 1500 else ""))

# ─────────────────────────────────────────────
# ANALYZE
# ─────────────────────────────────────────────
if st.button("⚡ Analyze Document", type="primary"):

    if not text.strip():
        st.warning("Please provide text or upload a PDF first.")
        st.stop()

    if len(text.split()) < 50:
        st.error("Document is too short to analyze meaningfully (need at least 50 words).")
        st.stop()

    st.markdown("---")

    with st.spinner("Running NLP analysis..."):
        top_keywords = extract_keywords(text, n=20)
        scored_sents = score_sentences(text, top_keywords)
        summary = diverse_summary(scored_sents, n=5)
        clauses = detect_clauses(text)
        risks = detect_risks(text)

    # ── KEYWORDS + WORDCLOUD ──────────────────
    st.markdown('<div class="section-header">🔑 Top Keywords</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1.3])

    with col1:
        df_kw = pd.DataFrame(top_keywords[:15], columns=["Keyword", "TF-IDF Score"])
        df_kw["Keyword"] = df_kw["Keyword"].str.upper()
        df_kw["TF-IDF Score"] = (df_kw["TF-IDF Score"] * 100).round(3)
        df_kw.index += 1
        st.dataframe(df_kw, use_container_width=True)

    with col2:
        st.pyplot(render_wordcloud(text))
        plt.close('all')

    # ── SMART SUMMARY ────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">📝 Smart Summary</div>', unsafe_allow_html=True)

    if summary:
        for i, sent in enumerate(summary, 1):
            st.markdown(f'<div class="highlight-box"><b>{i}.</b> {sent}</div>',
                        unsafe_allow_html=True)
    else:
        st.warning("Could not extract meaningful summary sentences. Try a longer document.")

    # ── CLAUSE DETECTION ─────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">📂 Detected Clauses</div>', unsafe_allow_html=True)

    if clauses:
        for cat, sentence in clauses.items():
            with st.expander(f"📌 {cat}"):
                st.markdown(f'<div class="highlight-box">{sentence}</div>', unsafe_allow_html=True)
    else:
        st.info("No specific clauses detected.")

    # ── RISK ANALYSIS ────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">⚠️ Risk Analysis</div>', unsafe_allow_html=True)

    r_col1, r_col2, r_col3 = st.columns(3)
    with r_col1:
        st.markdown("🔴 **High Risk**")
        if risks["high"]:
            for item in risks["high"]:
                st.markdown(f'<span class="tag risk-high">{item}</span>', unsafe_allow_html=True)
        else:
            st.markdown("_None detected_")

    with r_col2:
        st.markdown("🟠 **Medium Risk**")
        if risks["medium"]:
            for item in risks["medium"]:
                st.markdown(f'<span class="tag risk-medium">{item}</span>', unsafe_allow_html=True)
        else:
            st.markdown("_None detected_")

    with r_col3:
        st.markdown("🟢 **Low Risk**")
        if risks["low"]:
            for item in risks["low"]:
                st.markdown(f'<span class="tag risk-low">{item}</span>', unsafe_allow_html=True)
        else:
            st.markdown("_None detected_")

    # ── FREQUENCY CHART ──────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">📊 Keyword Frequency</div>', unsafe_allow_html=True)
    st.pyplot(render_freq_chart(text, top_keywords))
    plt.close('all')

    # ── DOCUMENT STATS ────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">📈 Document Statistics</div>', unsafe_allow_html=True)

    words_all = text.split()
    sentences_all = sent_tokenize(text)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Words", f"{len(words_all):,}")
    m2.metric("Sentences", f"{len(sentences_all):,}")
    m3.metric("Unique Words", f"{len(set(w.lower() for w in words_all)):,}")
    m4.metric("Avg Sentence Length", f"{len(words_all)//max(len(sentences_all),1)} words")