"""
SpamGuard A — High-Performance SMS Spam & Phishing Classifier
A modern, interactive, and visually stunning Streamlit dashboard powered by NLP,
TF-IDF n-grams, 14 structural numerical features, and Multinomial Naive Bayes.
"""

import os
import sys
import io
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import clean_text_for_nlp
from src.feature_extraction import extract_features_single, extract_features_batch, NUMERICAL_FEATURE_NAMES
from src.prediction import predict_sms, load_model_artifacts
from src.train_model import ensure_dataset, train_and_evaluate


# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & MODERN THEME
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SpamGuard | SMS Text Classifier",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom High-End Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Top Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        border-radius: 16px;
        padding: 24px 30px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 0%, #93c5fd 60%, #60a5fa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 6px 0;
        letter-spacing: -0.02em;
    }

    .hero-subtitle {
        font-size: 0.95rem;
        color: #94a3b8;
        font-weight: 400;
        margin: 0;
    }

    .kpi-chip-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 10px;
    }

    .kpi-chip {
        display: inline-flex;
        align-items: center;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15);
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.82rem;
        color: #e2e8f0;
        font-weight: 500;
    }

    .kpi-chip b {
        color: #ffffff;
        font-weight: 700;
        margin-left: 4px;
        margin-right: 4px;
    }

    /* Cards */
    .glass-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.03);
        margin-bottom: 16px;
    }

    /* Status Banners */
    .threat-banner-spam {
        background: linear-gradient(135deg, #fff1f2 0%, #fee2e2 100%);
        border: 2px solid #ef4444;
        border-radius: 14px;
        padding: 22px;
        box-shadow: 0 8px 20px rgba(239, 68, 68, 0.12);
        margin-bottom: 20px;
    }

    .threat-banner-ham {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 2px solid #22c55e;
        border-radius: 14px;
        padding: 22px;
        box-shadow: 0 8px 20px rgba(34, 197, 94, 0.12);
        margin-bottom: 20px;
    }

    .badge-tag {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    .badge-spam {
        background: #fee2e2;
        color: #dc2626;
        border: 1px solid #fca5a5;
    }

    .badge-ham {
        background: #dcfce7;
        color: #16a34a;
        border: 1px solid #86efac;
    }

    /* Stats Grid */
    .stat-tile {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 12px 14px;
        text-align: center;
    }

    .stat-tile-val {
        font-size: 1.35rem;
        font-weight: 700;
        color: #0f172a;
    }

    .stat-tile-lbl {
        font-size: 0.72rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        margin-top: 2px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. CACHED DATA & MODEL LOADERS
# -----------------------------------------------------------------------------
@st.cache_resource
def get_model_and_metadata():
    """Loads or trains the model artifacts in memory."""
    models_dir = "models"
    data_path = "data/SMSSpamCollection"
    model_path = os.path.join(models_dir, "spam_classifier.pkl")

    if not os.path.exists(model_path):
        with st.spinner("Initializing classifier on UCI SMS Spam dataset..."):
            ensure_dataset(data_path)
            train_and_evaluate(data_path, models_dir)

    model, vectorizer, metadata = load_model_artifacts(models_dir)
    return model, vectorizer, metadata


@st.cache_data
def get_dataset():
    """Loads and caches the cleaned dataset for fast visual analytics."""
    data_path = "data/SMSSpamCollection"
    ensure_dataset(data_path)
    df_raw = pd.read_csv(data_path, sep='\t', names=['label', 'message'], quoting=3, encoding='utf-8')
    df = df_raw.drop_duplicates(subset=['message']).reset_index(drop=True)
    
    df['message_length'] = df['message'].str.len()
    df['word_count'] = df['message'].str.split().str.len()
    df['digit_count'] = df['message'].apply(lambda x: sum(1 for c in str(x) if c.isdigit()))
    df['punct_count'] = df['message'].apply(lambda x: sum(1 for c in str(x) if c in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'))
    df['punct_ratio'] = df['punct_count'] / df['message_length'].replace(0, 1)
    df['upper_count'] = df['message'].apply(lambda x: sum(1 for c in str(x) if c.isupper()))
    df['alpha_count'] = df['message'].apply(lambda x: sum(1 for c in str(x) if c.isalpha()))
    df['cap_ratio'] = df['upper_count'] / df['alpha_count'].replace(0, 1)
    df['has_url'] = df['message'].str.contains(r'https?://|www\.|\.com|\.ly', regex=True).astype(int)
    df['has_currency'] = df['message'].str.contains(r'[\$£€₹]', regex=True).astype(int)
    
    return df_raw, df


def robust_read_csv(uploaded_file):
    """
    Ultra-robust CSV/text parser supporting:
    - Multiple encodings (UTF-8, UTF-8-SIG with Excel BOM, Latin-1, CP1252, ISO-8859-1)
    - Auto delimiter sniffing (comma, semicolon, tab, pipe)
    - Corrupt line skipping
    - Whitespace trimming on column headers
    """
    encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'iso-8859-1']
    for enc in encodings:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding=enc, sep=None, engine='python', on_bad_lines='skip')
            if df is not None and len(df.columns) > 0:
                df.columns = [str(c).strip() for c in df.columns]
                return df
        except Exception:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding=enc, on_bad_lines='skip')
                if df is not None and len(df.columns) > 0:
                    df.columns = [str(c).strip() for c in df.columns]
                    return df
            except Exception:
                continue

    # Raw fallback
    uploaded_file.seek(0)
    raw = uploaded_file.read().decode('utf-8', errors='replace')
    df = pd.read_csv(io.StringIO(raw), sep=None, engine='python', on_bad_lines='skip')
    df.columns = [str(c).strip() for c in df.columns]
    return df


def auto_detect_text_col(columns, df):
    """Finds the most likely text column automatically."""
    preferred = ['message', 'sms', 'text', 'v2', 'body', 'content', 'msg', 'sms_text', 'clean_message', 'email']
    col_map = {str(c).lower().strip(): c for c in columns}
    for p in preferred:
        if p in col_map:
            return col_map[p]
    
    # Fallback to column with highest average string length
    best_col = columns[0]
    max_len = -1
    for c in columns:
        try:
            avg_l = df[c].astype(str).str.len().mean()
            if avg_l > max_len:
                max_len = avg_l
                best_col = c
        except Exception:
            continue
    return best_col


# -----------------------------------------------------------------------------
# 3. MAIN CONTROLLER
# -----------------------------------------------------------------------------
def main():
    try:
        model, vectorizer, metadata = get_model_and_metadata()
    except Exception as e:
        st.error(f"Error initializing system: {e}")
        st.info("Execute `python src/train_model.py` in your terminal to initialize artifacts.")
        return

    metrics = metadata["metrics"]
    stats = metadata["dataset_stats"]

    # Hero Header Banner with clean spacing
    st.markdown(f"""
    <div class="hero-banner">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">
            <div style="display: flex; align-items: center; gap: 14px;">
                <svg width="44" height="44" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg" style="filter: drop-shadow(0 4px 10px rgba(56, 189, 248, 0.4)); flex-shrink: 0;">
                    <rect width="48" height="48" rx="14" fill="url(#logo_grad)" />
                    <path d="M24 11L14 15.2V22.5C14 28.6 18.3 34.3 24 35.8C29.7 34.3 34 28.6 34 22.5V15.2L24 11Z" fill="white" fill-opacity="0.95"/>
                    <path d="M22 25.2L18.5 21.8L17.2 23.2L22 28L30.8 19.2L29.5 17.8L22 25.2Z" fill="#1e3a8a"/>
                    <defs>
                        <linearGradient id="logo_grad" x1="0" y1="0" x2="48" y2="48" gradientUnits="userSpaceOnUse">
                            <stop stop-color="#38bdf8"/>
                            <stop offset="0.5" stop-color="#2563eb"/>
                            <stop offset="1" stop-color="#1d4ed8"/>
                        </linearGradient>
                    </defs>
                </svg>
                <div>
                    <h1 class="hero-title">SpamGuard</h1>
                    <p class="hero-subtitle">High-Precision SMS Text Spam & Phishing Classifier Powered by NLP and Naive Bayes</p>
                </div>
            </div>
            <div class="kpi-chip-container">
                <div class="kpi-chip">📊&nbsp;Corpus:<b>{stats['total_records_dedup']:,}</b>&nbsp;SMS</div>
                <div class="kpi-chip">🎯&nbsp;Accuracy:<b>{metrics['accuracy']*100:.1f}%</b></div>
                <div class="kpi-chip">⚡&nbsp;Precision:<b>{metrics['spam_precision']*100:.1f}%</b></div>
                <div class="kpi-chip">🔒&nbsp;100% Local Inference</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Clean Top-Level Tabs
    tab_inspect, tab_batch, tab_analytics, tab_arch = st.tabs([
        "⚡ Live SMS Classifier",
        "📁 Batch File Analyzer",
        "📊 Model & Dataset Analytics",
        "ℹ️ System Architecture"
    ])

    with tab_inspect:
        render_live_classifier(model, vectorizer, metadata)

    with tab_batch:
        render_batch_analyzer()

    with tab_analytics:
        render_analytics_dashboard(metadata)

    with tab_arch:
        render_architecture(metadata)


# -----------------------------------------------------------------------------
# TAB 1: LIVE SMS CLASSIFIER
# -----------------------------------------------------------------------------
def render_live_classifier(model, vectorizer, metadata):
    # Ensure textarea key is initialized in session state
    if "main_sms_textarea" not in st.session_state:
        st.session_state["main_sms_textarea"] = ""

    def set_live_scenario(msg):
        st.session_state["main_sms_textarea"] = msg

    def clear_live_text():
        st.session_state["main_sms_textarea"] = ""

    # Quick Scenario Chips (Clean, Rounded, One-click)
    st.markdown("##### ⚡ Quick-Test Scenarios:")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.button(
            "🎁 £1000 Prize Contest",
            use_container_width=True,
            on_click=set_live_scenario,
            args=("URGENT! You have won £1,000 cash. Click http://claim-prize.com to claim your reward before midnight!",)
        )
    with c2:
        st.button(
            "🏦 Bank Phishing Alert",
            use_container_width=True,
            on_click=set_live_scenario,
            args=("ALERT: Unusual sign-in attempt on your bank account. Verify your identity immediately: https://bit.ly/bank-security",)
        )
    with c3:
        st.button(
            "🔑 Legitimate Bank OTP",
            use_container_width=True,
            on_click=set_live_scenario,
            args=("Your Chase one-time verification code is 492015. Do not share this OTP with anyone for your security.",)
        )
    with c4:
        st.button(
            "💬 Friendly Casual Chat",
            use_container_width=True,
            on_click=set_live_scenario,
            args=("Hey! Are you free this afternoon to grab some coffee and review the data science lecture notes?",)
        )

    # Text Input Box
    user_text = st.text_area(
        "Enter or paste any SMS message to analyze:",
        height=100,
        placeholder="Type, paste, or click a scenario above...",
        key="main_sms_textarea"
    )

    btn_col1, btn_col2, _ = st.columns([2, 2, 6])
    with btn_col1:
        analyze_clicked = st.button("🚀 Analyze Message", type="primary", use_container_width=True)
    with btn_col2:
        st.button("🧹 Clear", use_container_width=True, on_click=clear_live_text)

    # Trigger Prediction
    if analyze_clicked or (user_text.strip() and user_text != ""):
        if not user_text.strip():
            st.warning("⚠️ Please enter a non-empty SMS text to classify.")
            return

        with st.spinner("Analyzing semantics, TF-IDF n-grams, and structural features..."):
            result = predict_sms(user_text)

        is_spam = result["label"] == "SPAM"
        spam_p = result["spam_probability"] * 100
        ham_p = result["ham_probability"] * 100
        feats = result["features"]

        st.markdown("---")

        # Two-Column Layout: Threat Assessment Banner + Animated Speedometer Gauge
        col_banner, col_gauge = st.columns([3, 2])

        with col_banner:
            if is_spam:
                st.markdown(f"""
                <div class="threat-banner-spam">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span class="badge-tag badge-spam">🚨 THREAT DETECTED</span>
                        <span style="font-size: 0.85rem; font-weight: 700; color: #dc2626;">Confidence: {result['confidence']*100:.1f}%</span>
                    </div>
                    <h2 style="color: #991b1b; margin: 8px 0 4px 0; font-size: 1.7rem; font-weight: 800;">
                        SPAM / FRAUDULENT MESSAGE
                    </h2>
                    <p style="color: #7f1d1d; font-size: 0.92rem; margin-bottom: 12px; line-height: 1.5;">
                        This message exhibits distinct signatures of promotional solicitation, monetary claims, or phishing attempts.
                    </p>
                    <div style="display: flex; gap: 24px;">
                        <div>
                            <div style="font-size: 0.72rem; text-transform: uppercase; color: #991b1b; font-weight: 700;">Spam Probability</div>
                            <div style="font-size: 1.6rem; font-weight: 800; color: #dc2626;">{spam_p:.1f}%</div>
                        </div>
                        <div>
                            <div style="font-size: 0.72rem; text-transform: uppercase; color: #64748b; font-weight: 700;">Ham Probability</div>
                            <div style="font-size: 1.6rem; font-weight: 800; color: #64748b;">{ham_p:.1f}%</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="threat-banner-ham">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span class="badge-tag badge-ham">✅ VERIFIED SAFE</span>
                        <span style="font-size: 0.85rem; font-weight: 700; color: #16a34a;">Confidence: {result['confidence']*100:.1f}%</span>
                    </div>
                    <h2 style="color: #15803d; margin: 8px 0 4px 0; font-size: 1.7rem; font-weight: 800;">
                        HAM / LEGITIMATE MESSAGE
                    </h2>
                    <p style="color: #14532d; font-size: 0.92rem; margin-bottom: 12px; line-height: 1.5;">
                        This message appears normal, reflecting standard interpersonal, educational, or transactional communication.
                    </p>
                    <div style="display: flex; gap: 24px;">
                        <div>
                            <div style="font-size: 0.72rem; text-transform: uppercase; color: #16a34a; font-weight: 700;">Ham Probability</div>
                            <div style="font-size: 1.6rem; font-weight: 800; color: #16a34a;">{ham_p:.1f}%</div>
                        </div>
                        <div>
                            <div style="font-size: 0.72rem; text-transform: uppercase; color: #64748b; font-weight: 700;">Spam Probability</div>
                            <div style="font-size: 1.6rem; font-weight: 800; color: #64748b;">{spam_p:.1f}%</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            if result["indicators"]:
                st.markdown("##### 🚩 Detected Heuristics:")
                for ind in result["indicators"]:
                    st.markdown(f"- ⚠️ **{ind}**")
            else:
                st.markdown("<p style='color: #15803d; font-size: 0.9rem;'>✓ No suspicious structural anomalies (URLs, currency symbols, excessive capitals) detected.</p>", unsafe_allow_html=True)

        with col_gauge:
            # High-Definition Plotly Arc Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=spam_p,
                number={'suffix': "%", 'font': {'size': 32, 'color': '#ef4444' if is_spam else '#16a34a'}},
                title={'text': "<b>Spam Threat Index</b>", 'font': {'size': 13, 'color': '#64748b'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#cbd5e1"},
                    'bar': {'color': "#ef4444" if is_spam else "#16a34a", 'thickness': 0.28},
                    'bgcolor': "#f8fafc",
                    'borderwidth': 1,
                    'bordercolor': "#cbd5e1",
                    'steps': [
                        {'range': [0, 30], 'color': '#dcfce7'},
                        {'range': [30, 70], 'color': '#fef3c7'},
                        {'range': [70, 100], 'color': '#fee2e2'}
                    ]
                }
            ))
            fig_gauge.update_layout(height=230, margin=dict(l=20, r=20, t=30, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

        # 14 Structural Feature Profile Tiles
        st.markdown("##### 📊 Extracted Structural Features:")
        f1, f2, f3, f4, f5, f6 = st.columns(6)
        with f1:
            st.markdown(f"""<div class="stat-tile"><div class="stat-tile-val">{int(feats['message_length'])}</div><div class="stat-tile-lbl">Characters</div></div>""", unsafe_allow_html=True)
        with f2:
            st.markdown(f"""<div class="stat-tile"><div class="stat-tile-val">{int(feats['word_count'])}</div><div class="stat-tile-lbl">Words</div></div>""", unsafe_allow_html=True)
        with f3:
            st.markdown(f"""<div class="stat-tile"><div class="stat-tile-val">{feats['capitalization_ratio']*100:.1f}%</div><div class="stat-tile-lbl">Capitalization</div></div>""", unsafe_allow_html=True)
        with f4:
            st.markdown(f"""<div class="stat-tile"><div class="stat-tile-val">{int(feats['digit_count'])}</div><div class="stat-tile-lbl">Digits</div></div>""", unsafe_allow_html=True)
        with f5:
            st.markdown(f"""<div class="stat-tile"><div class="stat-tile-val">{int(feats['currency_count'])}</div><div class="stat-tile-lbl">Currency ($ £ €)</div></div>""", unsafe_allow_html=True)
        with f6:
            st.markdown(f"""<div class="stat-tile"><div class="stat-tile-val">{'Yes' if feats['contains_url'] > 0 else 'No'}</div><div class="stat-tile-lbl">URL Detected</div></div>""", unsafe_allow_html=True)

        st.write("")

        # Deep Inspection & Export
        with st.expander("🔬 View NLP Preprocessing & Download JSON Diagnostic Report"):
            st.write(f"**Cleaned & Stemmed Tokens (TF-IDF):** `{result['cleaned_text']}`")
            report_data = {
                "input_message": user_text,
                "classification": result["label"],
                "spam_probability": result["spam_probability"],
                "ham_probability": result["ham_probability"],
                "indicators": result["indicators"],
                "features": feats
            }
            st.download_button(
                label="📥 Download Diagnostic Report (JSON)",
                data=json.dumps(report_data, indent=2),
                file_name="spamguard_diagnostic.json",
                mime="application/json"
            )


# -----------------------------------------------------------------------------
# TAB 2: BATCH FILE ANALYZER (BUG-FREE & STATE-PERSISTED)
# -----------------------------------------------------------------------------
def render_batch_analyzer():
    st.markdown("#### 📁 Batch SMS Classifier")
    st.caption("Upload any CSV or text file containing SMS messages, or paste multiple messages to classify all of them at once.")

    # Initialize batch state
    if "batch_results_df" not in st.session_state:
        st.session_state["batch_results_df"] = None

    sub_tab1, sub_tab2 = st.tabs(["📤 Upload CSV File", "📋 Paste Multiple Lines"])

    with sub_tab1:
        uploaded_file = st.file_uploader("Upload CSV / TXT file:", type=["csv", "txt"], key="batch_file_uploader")
        
        if uploaded_file is not None:
            try:
                raw_df = robust_read_csv(uploaded_file)
                
                if raw_df is None or raw_df.empty:
                    st.error("Uploaded file appears to be empty or unreadable.")
                else:
                    st.success(f"✓ Successfully read **{len(raw_df):,}** rows and **{len(raw_df.columns)}** columns.")
                    
                    # Preview first 3 rows
                    with st.expander("👁️ Preview First 3 Rows of File", expanded=True):
                        st.dataframe(raw_df.head(3), use_container_width=True)

                    # Auto-detect best SMS column
                    cols = list(raw_df.columns)
                    default_col = auto_detect_text_col(cols, raw_df)
                    default_idx = cols.index(default_col) if default_col in cols else 0

                    c_select, c_btn = st.columns([3, 2])
                    with c_select:
                        col_sel = st.selectbox(
                            "Select SMS Text Column to Classify:",
                            cols,
                            index=default_idx,
                            help="The column containing message strings."
                        )
                    with c_btn:
                        st.write("")
                        st.write("")
                        run_batch_clicked = st.button("⚡ Run Batch Classification", type="primary", use_container_width=True)

                    if run_batch_clicked:
                        with st.spinner(f"Classifying {len(raw_df):,} messages through NLP and Naive Bayes..."):
                            texts = raw_df[col_sel].fillna("").astype(str).tolist()
                            
                            # Fast inference
                            preds = [predict_sms(t) for t in texts]
                            
                            processed_df = raw_df.copy()
                            processed_df["Prediction"] = [
                                "HAM (Empty)" if p.get("error") else p["label"] for p in preds
                            ]
                            processed_df["Spam_Risk_%"] = [
                                0.0 if p.get("error") else round(p["spam_probability"] * 100, 1) for p in preds
                            ]
                            processed_df["Ham_Confidence_%"] = [
                                100.0 if p.get("error") else round(p["ham_probability"] * 100, 1) for p in preds
                            ]

                            # Save to session state so it persists across reruns!
                            st.session_state["batch_results_df"] = processed_df
                            st.rerun()

            except Exception as e:
                st.error(f"Error parsing uploaded file: {e}")
                st.info("Tip: Make sure your CSV file is properly formatted with headers.")

    with sub_tab2:
        pasted = st.text_area(
            "Paste messages (one message per line):",
            height=130,
            placeholder="Congratulations! You won £1000 prize. Call now\nAre we meeting for lunch today?\nURGENT: Verify your bank login at bit.ly/bank-auth\nCan you send over the homework slides?",
            key="pasted_batch_textarea"
        )
        if st.button("⚡ Classify Pasted Lines", type="primary"):
            lines = [l.strip() for l in pasted.split("\n") if l.strip()]
            if lines:
                with st.spinner(f"Classifying {len(lines)} messages..."):
                    preds = [predict_sms(l) for l in lines]
                    paste_df = pd.DataFrame({
                        "Message": lines,
                        "Prediction": ["HAM (Empty)" if p.get("error") else p["label"] for p in preds],
                        "Spam_Risk_%": [0.0 if p.get("error") else round(p["spam_probability"]*100, 1) for p in preds],
                        "Ham_Confidence_%": [100.0 if p.get("error") else round(p["ham_probability"]*100, 1) for p in preds]
                    })
                    st.session_state["batch_results_df"] = paste_df
                    st.rerun()
            else:
                st.warning("Please paste at least one line of text.")

    # Render Persisted Results
    res_df = st.session_state["batch_results_df"]
    if res_df is not None and not res_df.empty:
        st.markdown("---")
        total = len(res_df)
        spams = int((res_df["Prediction"] == "SPAM").sum())
        hams = total - spams
        spam_pct = (spams / total) * 100 if total > 0 else 0
        ham_pct = (hams / total) * 100 if total > 0 else 0

        # Batch KPI Cards
        st.markdown("##### 📊 Batch Classification Results:")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Total Messages", f"{total:,}")
        b2.metric("🚨 Spam Identified", f"{spams:,} ({spam_pct:.1f}%)")
        b3.metric("✅ Ham Identified", f"{hams:,} ({ham_pct:.1f}%)")
        b4.metric("Avg Spam Risk", f"{res_df['Spam_Risk_%'].mean():.1f}%")

        col_donut, col_data = st.columns([1, 2])
        with col_donut:
            fig_batch_pie = px.pie(
                names=["HAM (Safe)", "SPAM (Threat)"],
                values=[hams, spams],
                color=["HAM (Safe)", "SPAM (Threat)"],
                color_discrete_map={"HAM (Safe)": "#22c55e", "SPAM (Threat)": "#ef4444"},
                hole=0.45,
                title=f"Class Breakdown ({total} Total)"
            )
            fig_batch_pie.update_layout(height=290, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_batch_pie, use_container_width=True)

        with col_data:
            # Filter Radio
            filter_mode = st.radio(
                "Filter View:",
                [f"All Messages ({total})", f"🚨 Spam Only ({spams})", f"✅ Legitimate Only ({hams})"],
                horizontal=True
            )

            filtered_df = res_df
            if "Spam Only" in filter_mode:
                filtered_df = res_df[res_df["Prediction"] == "SPAM"]
            elif "Legitimate Only" in filter_mode:
                filtered_df = res_df[res_df["Prediction"] != "SPAM"]

            st.dataframe(filtered_df, use_container_width=True, height=240)

        # Download & Reset Row
        c_dl, c_reset, _ = st.columns([2, 2, 6])
        with c_dl:
            csv_bytes = res_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Labeled CSV",
                data=csv_bytes,
                file_name="classified_sms_batch.csv",
                mime="text/csv",
                key="btn_download_batch_csv"
            )
        with c_reset:
            if st.button("🗑️ Clear Batch Results", use_container_width=True):
                st.session_state["batch_results_df"] = None
                st.rerun()


# -----------------------------------------------------------------------------
# TAB 3: MODEL & DATASET ANALYTICS
# -----------------------------------------------------------------------------
def render_analytics_dashboard(metadata):
    _, df = get_dataset()
    metrics = metadata["metrics"]
    cm_dict = metadata["confusion_matrix"]
    cm = np.array(cm_dict["matrix"])

    # High-Impact Performance Metrics Row
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Validation Accuracy", f"{metrics['accuracy']*100:.2f}%")
    m2.metric("Spam Precision", f"{metrics['spam_precision']*100:.2f}%")
    m3.metric("Spam Recall", f"{metrics['spam_recall']*100:.2f}%")
    m4.metric("Spam F1-Score", f"{metrics['spam_f1']*100:.2f}%")
    m5.metric("ROC AUC", f"{metrics['roc_auc']:.4f}")

    st.markdown("---")

    # Row 1: Confusion Matrix & Dataset Donut
    c_cm, c_dist = st.columns(2)

    with c_cm:
        st.subheader("Confusion Matrix (1,035 Test Samples)")
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted Label", y="True Label", color="Count"),
            x=["Predicted HAM", "Predicted SPAM"],
            y=["Actual HAM", "Actual SPAM"],
            color_continuous_scale="Blues",
            text_auto=True
        )
        fig_cm.update_layout(height=320, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_cm, use_container_width=True)
        st.caption(f"✓ Only **{cm_dict['fp']} False Positives** out of {cm_dict['tn']+cm_dict['fp']} legitimate test messages (0.44% false alarm rate).")

    with c_dist:
        st.subheader("Dataset Class Distribution (5,169 Unique)")
        counts = df['label'].value_counts()
        fig_donut = px.pie(
            names=["HAM (Legitimate)", "SPAM (Threat)"],
            values=[counts['ham'], counts['spam']],
            color=["HAM (Legitimate)", "SPAM (Threat)"],
            color_discrete_map={"HAM (Legitimate)": "#22c55e", "SPAM (Threat)": "#ef4444"},
            hole=0.45
        )
        fig_donut.update_layout(height=320, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_donut, use_container_width=True)
        st.caption(f"Corpus: **{counts['ham']:,} HAM** ({counts['ham']/len(df)*100:.1f}%) and **{counts['spam']:,} SPAM** ({counts['spam']/len(df)*100:.1f}%).")

    st.markdown("---")

    # Row 2: Dual Curves (ROC & Precision-Recall)
    c_roc, c_pr = st.columns(2)
    roc_data = metadata["roc_curve"]
    pr_data = metadata["pr_curve"]

    with c_roc:
        st.subheader("ROC Curve (AUC = 0.9936)")
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=roc_data["fpr"], y=roc_data["tpr"], mode='lines', name="ROC", line=dict(color='#2563eb', width=2.5)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name="Chance", line=dict(dash='dash', color='#94a3b8')))
        fig_roc.update_layout(height=300, xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_roc, use_container_width=True)

    with c_pr:
        st.subheader("Precision-Recall Curve (AP = 0.9859)")
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=pr_data["recall"], y=pr_data["precision"], mode='lines', name="PR Curve", line=dict(color='#dc2626', width=2.5)))
        fig_pr.update_layout(height=300, xaxis_title="Recall", yaxis_title="Precision", margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_pr, use_container_width=True)

    st.markdown("---")

    # Row 3: Top Discriminative Terms
    st.subheader("Top Discriminative Vocabulary Learned by Model")
    top_terms = metadata.get("top_terms", {})
    col_t1, col_t2 = st.columns(2)

    with col_t1:
        if "spam" in top_terms:
            spam_df = pd.DataFrame(top_terms["spam"][:10], columns=["Keyword", "Log Probability"])
            fig_st = px.bar(spam_df[::-1], x='Log Probability', y='Keyword', orientation='h', color_discrete_sequence=['#ef4444'], title="Top Words in SPAM")
            fig_st.update_layout(height=340, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_st, use_container_width=True)

    with col_t2:
        if "ham" in top_terms:
            ham_df = pd.DataFrame(top_terms["ham"][:10], columns=["Keyword", "Log Probability"])
            fig_ht = px.bar(ham_df[::-1], x='Log Probability', y='Keyword', orientation='h', color_discrete_sequence=['#22c55e'], title="Top Words in HAM")
            fig_ht.update_layout(height=340, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_ht, use_container_width=True)


# -----------------------------------------------------------------------------
# TAB 4: SYSTEM ARCHITECTURE
# -----------------------------------------------------------------------------
def render_architecture(metadata):
    st.markdown("#### 🏗️ System Architecture & Dual-Stream Pipeline")

    st.markdown("""
    ```text
    Incoming SMS Message
          │
          ├───────────────────────────────────┬────────────────────────────────────┐
          ▼                                   ▼                                    ▼
    NLP Text Normalization            Structural Feature Extraction         Label Stratification
    • Lowercase conversion            • Message & Word Length               • 87% Ham / 13% Spam
    • URL & HTML stripping            • Uppercase & Punctuation Ratios      • Prevents data leakage
    • NLTK Stopword elimination       • Digit Density & Counts
    • Porter Stemming (win, call)     • Currency Symbols ($ £ € ₹)
          │                           • Exclamations & Question Marks
          ▼                                   │
    TF-IDF Vectorizer                         ▼
    • (1, 2) Unigram + Bigrams        MinMaxScaler [0, 1]
    • Sublinear TF Scaling                    │
          │                                   │
          └─────────────────┬─────────────────┘
                            ▼
                  Sparse Feature Fusion
                 (TF-IDF + Scaled Ratios)
                            │
                            ▼
                 Multinomial Naive Bayes
                 (Laplace Smoothing α=0.2)
                            │
                            ▼
              Exact Posterior Probabilities
                P(Spam|X)  vs  P(Ham|X)
                            │
                            ▼
                     Decision Output
    ```
    """)

    st.markdown("""
    ##### Key Advantages of this Architecture:
    1. **Adversarial Resilience:** Keyword misspellings (`w1n`, `fr33`) fail to evade the model because structural anomalies (capitalization density, digits, currency symbols) trigger the classifier.
    2. **Low Latency (<5ms):** Naive Bayes evaluates posterior probabilities via logarithmic sums, ensuring near-instant predictions on standard CPUs.
    3. **Privacy-Preserving:** Local inference runs entirely on the host machine without streaming user texts to third-party APIs.
    """)


if __name__ == "__main__":
    main()
