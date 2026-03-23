import streamlit as st
import requests
import pandas as pd
from pathlib import Path
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="PolitiCheck 2024 | Research Portal",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ADVANCED CUSTOM CSS (Premium Research Theme) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=Montserrat:wght@300;400;600;800&display=swap');

    /* Global Overrides */
    .main {
        background-color: #f4f6f9;
        font-family: 'Montserrat', sans-serif;
    }
    
    /* Elegant Headers */
    h1, h2, h3 {
        color: #0d1b2a !important;
        font-family: 'Libre Baskerville', serif;
        font-weight: 700;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0d1b2a !important;
        color: #e0e1dd !important;
    }
    section[data-testid="stSidebar"] .stMarkdown p {
        color: #e0e1dd !important;
    }

    /* Modern Card Design */
    .res-card {
        background: white;
        padding: 30px;
        border-radius: 15px;
        border-left: 6px solid #1b263b;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        margin-bottom: 25px;
        transition: transform 0.2s ease;
    }
    .res-card:hover {
        transform: translateY(-5px);
    }

    /* Verdict Banners */
    .verdict-banner {
        font-family: 'Libre Baskerville', serif;
        font-size: 28px;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 30px;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .v-likely { background: linear-gradient(135deg, #2d6a4f, #40916c); }
    .v-partial { background: linear-gradient(135deg, #d97706, #f59e0b); }
    .v-unlikely { background: linear-gradient(135deg, #991b1b, #dc2626); }

    /* Button Styling */
    .stButton>button {
        background-color: #1b263b !important;
        color: #e0e1dd !important;
        border: none !important;
        padding: 15px 30px !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        width: 100%;
        font-size: 16px !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton>button:hover {
        background-color: #415a77 !important;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1) !important;
    }

    /* Metrics and Icons */
    .metric-val {
        font-size: 32px;
        font-weight: 800;
        color: #1b263b;
    }
    .metric-label {
        font-size: 14px;
        color: #778da9;
        text-transform: uppercase;
    }
    </style>
    """, unsafe_allow_html=True)

# --- API ENDPOINT ---
API_URL = "http://127.0.0.1:8000/predict"

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<h1 style='color:#e0e1dd; text-align:center;'>🏛️ PolitiCheck</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-style:italic;'>Semantic Governance Intelligence</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("### 🔍 Model Configuration")
    use_llm = st.toggle("Llama-3.3 Skeptic Layer", value=True)
    st.caption("Cross-references semantic similarity with socio-economic feasibility data.")
    
    st.markdown("---")
    st.markdown("### 📊 Dataset Integrity")
    st.markdown("""
    - **Archives:** 2014 & 2019 Mapped
    - **NLP Engine:** SBERT L6-v2
    - **Gold DB Size:** ~1,000 Entries
    """)
    st.success("Core Engine: Ready")
    
    st.markdown("---")
    st.markdown("<p style='font-size:10px; opacity:0.6;'>RESEARCH VERSION 2.0.4<br>COMPLIANT WITH ACADEMIC SECTIONS 5 & 6</p>", unsafe_allow_html=True)

# --- HEADER SECTION ---
st.markdown("<h1 style='text-align:center; font-size: 52px; margin-bottom: 0;'>MANIFESTO ANALYSIS PORTAL</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size: 16px; color: #778da9; letter-spacing: 2px; text-transform: uppercase;'>Evidence-Based Political Outcome Prediction</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# --- MAIN INTERFACE ---
with st.container():
    st.markdown("<div class='res-card'>", unsafe_allow_html=True)
    st.markdown("### 📝 Analysis Input")
    promise_input = st.text_area("Input promise from 2024 Manifesto for historical benchmarking:", 
                                placeholder="Paste the text here (e.g., 'We will double the budget for infrastructure...')",
                                height=100,
                                label_visibility="collapsed")
    
    col_b1, col_b2, col_b3 = st.columns([1, 1, 1])
    with col_b2:
        predict_button = st.button("RUN DEEP SEMANTIC AUDIT")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- LOGIC ---
if predict_button:
    if not promise_input.strip():
        st.error("Protocol Error: Input text cannot be null.")
    else:
        with st.spinner("Decoding Semantic Vectors..."):
            try:
                # Add a tiny delay for visual pacing
                time.sleep(0.5)
                response = requests.post(API_URL, json={"text": promise_input, "use_llm": use_llm})
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # --- BANNER ---
                    verdict = data['final_verdict']
                    if "Likely" in verdict: v_cls, v_label = "v-likely", "HIGH PROBABILITY OF FULFILLMENT"
                    elif "Partial" in verdict: v_cls, v_label = "v-partial", "POTENTIAL PARTIAL IMPLEMENTATION"
                    else: v_cls, v_label = "v-unlikely", "LOW PROBABILITY / HISTORICAL FRICTION"
                    
                    st.markdown(f'<div class="verdict-banner {v_cls}">{v_label}</div>', unsafe_allow_html=True)
                    
                    # --- RESULTS GRID ---
                    c_main, c_side = st.columns([3, 2])
                    
                    with c_main:
                        # Semantic Analysis Card
                        st.markdown('<div class="res-card">', unsafe_allow_html=True)
                        st.markdown("### 📊 Semantic Precedent Report")
                        st.write(data['semantic_analysis']['reasoning'])
                        
                        m1, m2, m3 = st.columns(3)
                        m1.markdown(f'<div class="metric-label">Confidence</div><div class="metric-val">{data["semantic_analysis"]["confidence"]*100:.1f}%</div>', unsafe_allow_html=True)
                        m2.markdown(f'<div class="metric-label">Engine</div><div class="metric-val">SBERT</div>', unsafe_allow_html=True)
                        m3.markdown(f'<div class="metric-label">Source</div><div class="metric-val">Archive</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Skeptic Card
                        if use_llm and data.get('llm_review'):
                            st.markdown('<div class="res-card" style="border-left-color: #0d1b2a; background-color: #f8f9fa;">', unsafe_allow_html=True)
                            st.markdown("### 🤖 Skeptic AI Audit (Llama-3.3-70b)")
                            st.markdown(f"<p style='font-style: italic; color: #415a77;'>\"{data['llm_review']['llm_reasoning']}\"</p>", unsafe_allow_html=True)
                            
                            st.markdown("#### Audit Factors:")
                            f_cols = st.columns(len(data['llm_review'].get('key_factors', [])))
                            for idx, factor in enumerate(data['llm_review'].get('key_factors', [])):
                                f_cols[idx].markdown(f"✅ **{factor}**")
                            
                            if data['llm_review'].get('repeat_promise'):
                                st.error("🚩 WARNING: Historical Repetition detected. High implementation risk.")
                            st.markdown('</div>', unsafe_allow_html=True)

                    with c_side:
                        # Evidence Panel
                        st.markdown("### 📂 Evidence Dossier")
                        for idx, match in enumerate(data['historical_evidence']):
                            with st.expander(f"Historical Match #{idx+1} (Score: {match['similarity']:.2f})"):
                                st.markdown(f"**Archive:** {match['year']} BJP Review")
                                st.markdown(f"**Historical Outcome:** `{match['outcome']}`")
                                st.markdown(f"---")
                                st.markdown(f"<div style='font-size: 13px; color: #415a77;'>{match['historical_text']}</div>", unsafe_allow_html=True)
                        
                        # Discrepancy Alert
                        if data.get('discrepancy'):
                            st.warning(f"**Audit Discrepancy:** {data['discrepancy_note']}")

                else:
                    st.error(f"Backend Server Error: {response.status_code}")
            except Exception as e:
                st.error("CRITICAL: Backend API Offline. Ensure 'uvicorn main:app' is active on port 8000.")

# --- FOOTER ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; border-top: 1px solid #e0e1dd; padding-top: 20px; color: #778da9;'>
        <b>POLITICHECK RESEARCH GROUP</b> | Semantic NLP & Governance Track Record Prediction<br>
        Built for Academic Submission - 2024-25
    </div>
    """, unsafe_allow_html=True)
