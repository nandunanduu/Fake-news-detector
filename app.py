"""
Fake News Detector — Professional UI
Run:  python3 -m streamlit run app.py
"""
import os, sys
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))
from src.model import FakeNewsDetector
from src.preprocessing import clean_text
from src.utils import save_prediction, get_history_df, clear_history, fetch_article_from_url

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Google Fonts + CSS ─────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Inter:wght@300;400;500;600&family=Space+Grotesk:wght@400;600&display=swap" rel="stylesheet">

<style>
/* ══════════════════════════════════════════════════════
   DESIGN TOKENS
   We lean on Streamlit's own CSS variables so the UI
   automatically adapts when the user toggles dark/light.
   --background-color            → page bg
   --secondary-background-color  → sidebar / card bg
   --text-color                  → body text
   ══════════════════════════════════════════════════════ */
:root {
  --fn-accent:  #00d4ff;
  --fn-accent2: #6366f1;
  --fn-accent3: #a855f7;
  --fn-fake:    #ef4444;
  --fn-real:    #10b981;
  --fn-border:  rgba(99,116,143,0.22);
  --fn-muted:   #6b7a99;
}

/* ── Base — let Streamlit own the backgrounds ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
/* DO NOT set .stApp background — that's what breaks light mode */
section[data-testid="stSidebar"] { border-right: 1px solid var(--fn-border); }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-thumb { background: var(--fn-accent2); border-radius: 3px; }

/* ══ ANIMATIONS ══ */
@keyframes gradientShift {
  0%   { background-position: 0% 50%; }
  50%  { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}
@keyframes glowPulse {
  0%,100% { filter: drop-shadow(0 0 12px rgba(0,212,255,.45)) drop-shadow(0 0 28px rgba(99,102,241,.25)); }
  50%     { filter: drop-shadow(0 0 22px rgba(0,212,255,.75)) drop-shadow(0 0 48px rgba(99,102,241,.45)); }
}
@keyframes fadeSlideDown {
  from { opacity:0; transform:translateY(-16px); }
  to   { opacity:1; transform:translateY(0); }
}
@keyframes scanLine {
  0%   { transform:translateX(-100%); }
  100% { transform:translateX(400%); }
}
@keyframes verdictIn {
  from { opacity:0; transform:scale(.96) translateY(8px); }
  to   { opacity:1; transform:scale(1) translateY(0); }
}

/* ── Hero ── */
.hero-wrapper {
  text-align:center; padding:40px 0 26px;
  animation:fadeSlideDown .65s ease both;
}
.hero-title {
  font-family:'Orbitron',sans-serif; font-weight:900;
  font-size:clamp(1.8rem,5vw,3.2rem);
  background:linear-gradient(270deg,#00d4ff,#6366f1,#a855f7,#00d4ff);
  background-size:300% 300%;
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
  animation:gradientShift 5s ease infinite, glowPulse 3s ease-in-out infinite;
  line-height:1.15; margin:0; position:relative; overflow:hidden;
}
.hero-title::after {
  content:''; position:absolute; top:0; left:0; width:25%; height:100%;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.1),transparent);
  animation:scanLine 3.5s ease-in-out infinite;
}
.hero-sub {
  font-family:'Space Grotesk',sans-serif; font-size:1rem;
  color:var(--fn-muted); margin-top:12px; letter-spacing:.3px;
}
.hero-sub span { color:var(--fn-accent); font-weight:600; }
.hero-divider {
  width:60px; height:3px;
  background:linear-gradient(90deg,var(--fn-accent),var(--fn-accent2));
  border-radius:2px; margin:14px auto 0;
}

/* ── Textarea — only touch what matters; let Streamlit theme control bg/text ── */
textarea {
  border-radius:10px !important;
  font-family:'Inter',sans-serif !important; font-size:.95rem !important;
  transition:border-color .2s,box-shadow .2s !important;
}
textarea:focus {
  border-color:var(--fn-accent) !important;
  box-shadow:0 0 0 2px rgba(0,212,255,.15) !important;
}

/* ── Analyse button ── */
button[kind="primary"] {
  background:linear-gradient(135deg,#0ea5e9,#6366f1) !important;
  border:none !important; border-radius:10px !important;
  font-family:'Space Grotesk',sans-serif !important; font-weight:600 !important;
  font-size:1rem !important; letter-spacing:.5px !important;
  transition:all .2s !important; color:#fff !important;
  box-shadow:0 4px 15px rgba(14,165,233,.3) !important;
}
button[kind="primary"]:hover {
  transform:translateY(-1px) !important;
  box-shadow:0 6px 22px rgba(14,165,233,.45) !important;
}

/* ── Secondary/clear button — theme-aware ── */
button[kind="secondary"] {
  border-radius:8px !important; font-size:.85rem !important;
  transition:all .2s !important; border:1px solid var(--fn-border) !important;
}

/* ── Verdict banners ── */
.verdict-fake {
  background:linear-gradient(135deg,#7f1d1d 0%,#dc2626 50%,#991b1b 100%);
  border:1px solid rgba(239,68,68,.5); border-radius:14px;
  padding:22px 32px; text-align:center; color:#fff;
  font-family:'Orbitron',sans-serif; font-size:clamp(1.2rem,3vw,1.7rem);
  font-weight:700; letter-spacing:2px;
  box-shadow:0 0 40px rgba(220,38,38,.35),inset 0 1px 0 rgba(255,255,255,.12);
  animation:verdictIn .4s ease both; margin-bottom:4px;
}
.verdict-real {
  background:linear-gradient(135deg,#064e3b 0%,#059669 50%,#065f46 100%);
  border:1px solid rgba(16,185,129,.5); border-radius:14px;
  padding:22px 32px; text-align:center; color:#fff;
  font-family:'Orbitron',sans-serif; font-size:clamp(1.2rem,3vw,1.7rem);
  font-weight:700; letter-spacing:2px;
  box-shadow:0 0 40px rgba(16,185,129,.35),inset 0 1px 0 rgba(255,255,255,.12);
  animation:verdictIn .4s ease both; margin-bottom:4px;
}

/* ── Probability cards — use Streamlit secondary bg so they adapt to theme ── */
.prob-card {
  background:var(--secondary-background-color);
  border:1px solid var(--fn-border); border-radius:12px;
  padding:18px; text-align:center;
  animation:verdictIn .5s .1s ease both; opacity:0; animation-fill-mode:forwards;
}
.prob-value-fake { font-size:2rem; font-weight:700; color:var(--fn-fake); font-family:'Orbitron',sans-serif; }
.prob-value-real { font-size:2rem; font-weight:700; color:var(--fn-real); font-family:'Orbitron',sans-serif; }
.prob-label { font-size:.78rem; color:var(--fn-muted); letter-spacing:1.5px; text-transform:uppercase; margin-top:4px; }

/* ── Section header ── */
.section-header {
  font-family:'Space Grotesk',sans-serif; font-size:1rem; font-weight:600;
  color:var(--fn-muted); letter-spacing:.5px; margin:26px 0 12px;
  display:flex; align-items:center; gap:8px;
}
.section-header::after {
  content:''; flex:1; height:1px;
  background:linear-gradient(90deg,var(--fn-border),transparent);
}

/* ── Sidebar logo ── */
.sidebar-logo {
  font-family:'Orbitron',sans-serif; font-size:.9rem; font-weight:700;
  background:linear-gradient(90deg,var(--fn-accent),var(--fn-accent2));
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
  letter-spacing:2px;
}
.sidebar-divider { border:none; border-top:1px solid var(--fn-border); margin:14px 0; }

/* ── Metrics — secondary bg ── */
[data-testid="stMetric"] {
  background:var(--secondary-background-color) !important;
  border:1px solid var(--fn-border) !important; border-radius:10px; padding:12px 16px;
}
[data-testid="stMetricValue"] { font-family:'Orbitron',sans-serif !important; font-size:1.4rem !important; }
[data-testid="stMetricLabel"] { font-size:.75rem !important; letter-spacing:1px !important; }

/* ── Misc ── */
[data-testid="stFileUploader"] { border:1px dashed var(--fn-border) !important; border-radius:10px !important; }
[data-testid="stDataFrame"]    { border:1px solid var(--fn-border); border-radius:10px; overflow:hidden; }
.stTextArea label { font-size:.85rem !important; letter-spacing:.5px !important; }
.news-label {
  font-size:.78rem; letter-spacing:1.5px; text-transform:uppercase;
  color:var(--fn-muted); margin-bottom:4px;
}

/* ── URL card ── */
.url-card {
  background:var(--secondary-background-color);
  border:1px solid var(--fn-border); border-radius:12px;
  padding:20px 22px; margin-bottom:4px;
}
.url-meta {
  background:var(--secondary-background-color);
  border-left:3px solid var(--fn-accent); border-radius:0 8px 8px 0;
  padding:10px 14px; margin-top:10px; font-size:.85rem;
}
.url-meta strong { color:var(--fn-accent); }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────
if 'news_text'    not in st.session_state: st.session_state['news_text']    = ''
if 'url_input'    not in st.session_state: st.session_state['url_input']    = ''
if 'url_fetched'  not in st.session_state: st.session_state['url_fetched']  = None  # dict or None
if 'last_result'  not in st.session_state: st.session_state.last_result     = None

def _clear_all():
    st.session_state['news_text']   = ''
    st.session_state['url_input']   = ''
    st.session_state['url_fetched'] = None
    st.session_state.last_result    = None

# ── Load model ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(mtype):
    try:
        return FakeNewsDetector.load(mtype)
    except FileNotFoundError:
        return None

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">⚙ FAKE NEWS DETECTOR</div>', unsafe_allow_html=True)
    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    st.markdown("**Model**")
    model_choice = st.radio(
        "", ["Logistic Regression", "Naive Bayes"],
        label_visibility="collapsed",
        help="Logistic Regression is generally more accurate on large datasets.",
    )
    model_key = 'logistic_regression' if model_choice == "Logistic Regression" else 'naive_bayes'

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # ── Prediction History ───────────────────────────────────────────────
    with st.expander("📋  Prediction History", expanded=False):
        df_hist = get_history_df()
        if df_hist.empty:
            st.caption("No predictions yet.")
        else:
            total  = len(df_hist)
            n_fake = (df_hist['prediction'] == 'FAKE').sum()
            n_real = total - n_fake

            c1, c2 = st.columns(2)
            c1.metric("🚨 FAKE", n_fake)
            c2.metric("✅ REAL", n_real)

            if total > 1:
                fig_pie, ax_pie = plt.subplots(figsize=(3, 2.5))
                fig_pie.patch.set_alpha(0)
                ax_pie.set_facecolor('none')
                ax_pie.pie(
                    [n_fake, n_real], labels=['FAKE', 'REAL'],
                    colors=['#ef4444', '#10b981'],
                    autopct='%1.0f%%', startangle=90,
                    textprops={'fontsize': 9},
                )
                plt.tight_layout()
                st.pyplot(fig_pie, use_container_width=True)
                plt.close(fig_pie)

            disp = df_hist.iloc[::-1].reset_index(drop=True)
            disp['confidence'] = disp['confidence'].map(lambda x: f"{x:.1f}%")

            def _color(val):
                return 'color:#ef4444;font-weight:600' if val == 'FAKE' else 'color:#10b981;font-weight:600'

            st.dataframe(
                disp.style.map(_color, subset=['prediction']),
                use_container_width=True, height=220,
            )
            csv = disp.to_csv(index=False).encode()
            st.download_button("⬇ Export CSV", csv, "history.csv", "text/csv", use_container_width=True)

        st.markdown("")
        if st.button("🗑  Clear History", use_container_width=True):
            clear_history()
            st.success("History cleared.")
            st.rerun()

    # ── Model Info ───────────────────────────────────────────────────────
    with st.expander("📊  Model Info", expanded=False):
        det_info = load_model(model_key)
        if det_info and det_info.metrics:
            m = det_info.metrics
            st.metric("Accuracy",  f"{m['accuracy']*100:.1f}%")
            st.metric("Precision", f"{m['precision']*100:.1f}%")
            st.metric("Recall",    f"{m['recall']*100:.1f}%")
            st.metric("F1 Score",  f"{m['f1']*100:.1f}%")

            cm = m['confusion_matrix']
            fig_cm, ax_cm = plt.subplots(figsize=(3.5, 2.8))
            fig_cm.patch.set_alpha(0)
            ax_cm.set_facecolor('none')
            sns.heatmap(
                cm, annot=True, fmt='d',
                xticklabels=['FAKE', 'REAL'], yticklabels=['FAKE', 'REAL'],
                cmap='RdYlGn', ax=ax_cm, linewidths=.5, cbar=False,
                annot_kws={'fontsize': 11},
            )
            ax_cm.set_xlabel("Predicted", fontsize=9)
            ax_cm.set_ylabel("Actual",    fontsize=9)
            ax_cm.tick_params(labelsize=9)
            for spine in ax_cm.spines.values(): spine.set_visible(False)
            plt.tight_layout()
            st.pyplot(fig_cm, use_container_width=True)
            plt.close(fig_cm)

            with st.expander("Full Report"):
                st.code(m['classification_report'], language=None)
        else:
            st.info("Run `python3 train.py` first.")

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.caption("TF-IDF · ML · v1.0")

# ── Active model ───────────────────────────────────────────────────────────
detector = load_model(model_key)

# ── Hero ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrapper">
  <h1 class="hero-title">FAKE NEWS DETECTOR</h1>
  <p class="hero-sub">AI-Powered · Detect <span>misinformation</span> instantly</p>
  <div class="hero-divider"></div>
</div>
""", unsafe_allow_html=True)

if detector is None:
    st.error("**No trained model found.** Run `python3 train.py` first, then refresh.", icon="⚠️")
    st.stop()

# ── Main predict panel ─────────────────────────────────────────────────────
col_main, _ = st.columns([1, 0.02])

with col_main:
    input_mode = st.radio(
        "Input method",
        ["✏️  Type / Paste", "🔗  Enter URL", "📂  Upload .txt file"],
        horizontal=True, label_visibility="collapsed",
    )

    news_text = ""

    # ── Option 1: Type / Paste ───────────────────────────────────────────
    if input_mode == "✏️  Type / Paste":
        col_label, col_clr = st.columns([5, 1])
        with col_label:
            st.markdown('<p class="news-label">News Article Text</p>', unsafe_allow_html=True)
        with col_clr:
            st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)
            st.button("✕ Clear", key="clear_btn", on_click=_clear_all)

        news_text = st.text_area(
            "news_text_area",
            height=190,
            placeholder="Paste a news headline or full article here …",
            label_visibility="collapsed",
            key="news_text",
        )

    # ── Option 2: URL ────────────────────────────────────────────────────
    elif input_mode == "🔗  Enter URL":
        st.markdown('<p class="news-label">News Article URL</p>', unsafe_allow_html=True)

        col_url, col_fetch = st.columns([5, 1])
        with col_url:
            url_val = st.text_input(
                "url_input_label",
                placeholder="https://www.example.com/news/article",
                label_visibility="collapsed",
                key="url_input",
            )
        with col_fetch:
            st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)
            fetch_clicked = st.button("⬇ Fetch", use_container_width=True)

        if fetch_clicked:
            if not url_val.strip():
                st.warning("Please enter a URL first.", icon="🔗")
            else:
                with st.spinner("Fetching article …"):
                    title, body, err = fetch_article_from_url(url_val.strip())
                if err:
                    st.error(err, icon="❌")
                    st.session_state['url_fetched'] = None
                else:
                    st.session_state['url_fetched'] = {'title': title, 'body': body, 'url': url_val.strip()}
                    st.session_state.last_result = None  # clear old result

        fetched = st.session_state.get('url_fetched')
        if fetched:
            news_text = fetched['body']
            # Metadata card
            word_count = len(news_text.split())
            title_disp = fetched['title'] or "No title detected"
            st.markdown(
                f'<div class="url-meta">'
                f'<strong>📰 {title_disp}</strong><br>'
                f'<span style="font-size:.8rem;opacity:.7">'
                f'{fetched["url"][:70]}{"…" if len(fetched["url"])>70 else ""}'
                f' &nbsp;·&nbsp; {word_count:,} words extracted</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            with st.expander("Preview extracted text", expanded=False):
                st.text(news_text[:1000] + ("…" if len(news_text) > 1000 else ""))

            st.markdown('<div style="height:2px"></div>', unsafe_allow_html=True)
            if st.button("✕  Clear URL", on_click=_clear_all, use_container_width=False):
                pass

    # ── Option 3: Upload .txt ────────────────────────────────────────────
    else:
        uploaded = st.file_uploader(
            "Upload .txt", type=["txt"], label_visibility="collapsed"
        )
        if uploaded:
            news_text = uploaded.read().decode("utf-8", errors="ignore")
            with st.expander("Preview file", expanded=True):
                st.text(news_text[:800] + ("…" if len(news_text) > 800 else ""))

    st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)
    analyse_clicked = st.button("🔬  Analyse Article", type="primary", use_container_width=True)

    # ── Run prediction ───────────────────────────────────────────────────
    if analyse_clicked:
        active_text = news_text or st.session_state.get('news_text', '')
        if not active_text.strip():
            st.warning("Please enter or upload some text first.", icon="💬")
        else:
            cleaned = clean_text(active_text)
            if len(cleaned.split()) < 3:
                st.warning("Text too short after preprocessing. Please add more content.", icon="⚠️")
            else:
                with st.spinner("Analysing …"):
                    prediction, confidence, prob_dict = detector.predict(cleaned)
                    top_words = detector.get_top_features(cleaned, n=15)
                    save_prediction(active_text, prediction, confidence, model_key)
                    st.session_state.last_result = {
                        'prediction': prediction,
                        'confidence': confidence,
                        'prob_dict':  prob_dict,
                        'top_words':  top_words,
                    }

    # ── Show results ─────────────────────────────────────────────────────
    result = st.session_state.last_result
    if result:
        prediction = result['prediction']
        confidence = result['confidence']
        prob_dict  = result['prob_dict']
        top_words  = result['top_words']

        st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

        # Verdict banner
        if prediction == 'FAKE':
            st.markdown(
                f'<div class="verdict-fake">🚨 &nbsp;FAKE NEWS &nbsp;|&nbsp; {confidence*100:.1f}% Confidence</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="verdict-real">✅ &nbsp;CREDIBLE NEWS &nbsp;|&nbsp; {confidence*100:.1f}% Confidence</div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)

        # Probability cards
        p_fake = prob_dict.get('FAKE', 0)
        p_real = prob_dict.get('REAL', 0)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f'<div class="prob-card"><div class="prob-value-fake">{p_fake*100:.1f}%</div>'
                f'<div class="prob-label">FAKE probability</div></div>',
                unsafe_allow_html=True,
            )
            st.progress(float(p_fake))
        with c2:
            st.markdown(
                f'<div class="prob-card"><div class="prob-value-real">{p_real*100:.1f}%</div>'
                f'<div class="prob-label">REAL probability</div></div>',
                unsafe_allow_html=True,
            )
            st.progress(float(p_real))

        # Word influence chart
        if top_words:
            st.markdown(
                '<div class="section-header">🔑 Key Words Influencing This Prediction</div>',
                unsafe_allow_html=True,
            )

            words  = [w for w, _, _ in top_words]
            scores = [s for _, s, _ in top_words]
            dirs   = [d for _, _, d in top_words]
            colors = ['#ef4444' if d == 'FAKE' else '#10b981' for d in dirs]

            fig, ax = plt.subplots(figsize=(9, max(3.2, len(words) * 0.42)))
            # Transparent background so it adapts to dark AND light mode
            fig.patch.set_alpha(0)
            ax.set_facecolor('none')

            y_pos = np.arange(len(words))
            ax.barh(y_pos, scores, color=colors, edgecolor='none', height=0.55)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(words, fontsize=10, fontfamily='monospace')
            ax.tick_params(axis='x', labelsize=8)
            ax.tick_params(axis='y', length=0)
            for spine in ax.spines.values():
                spine.set_edgecolor(mpatches.Patch().get_facecolor())  # invisible
                spine.set_alpha(0.2)
            ax.axvline(0, color='gray', linewidth=0.8, alpha=0.4)
            ax.set_xlabel(
                "Influence score  (negative → FAKE  ·  positive → REAL)",
                fontsize=8.5, alpha=0.6,
            )
            ax.set_xlim(
                min(scores) * 1.15 if min(scores) < 0 else -0.01,
                max(scores) * 1.15 if max(scores) > 0 else 0.01,
            )

            fake_p = mpatches.Patch(color='#ef4444', label='Pushes toward FAKE')
            real_p = mpatches.Patch(color='#10b981', label='Pushes toward REAL')
            ax.legend(
                handles=[fake_p, real_p], fontsize=8.5,
                framealpha=0.15, edgecolor='gray',
            )

            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
