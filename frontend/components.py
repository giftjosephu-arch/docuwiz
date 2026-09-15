"""
DocuWiz AI - Frontend UI Components & CSS Styling
Provides custom glassmorphism styles, metric badges, and cards for Streamlit.
"""

import streamlit as st


def inject_custom_css():
    """Inject modern styling into the Streamlit app."""
    st.markdown("""
    <style>
    /* Global Typography & Palette */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Main container padding */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
    }
    
    /* Header hero banner */
    .hero-container {
        background: linear-gradient(135deg, #1E1B4B 0%, #312E81 50%, #4338CA 100%);
        padding: 24px 30px;
        border-radius: 16px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(49, 46, 129, 0.25);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        opacity: 0.9;
        margin-top: 8px;
        font-weight: 400;
        line-height: 1.5;
    }
    .hero-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 14px;
    }
    .hero-tag {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(8px);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Custom metric cards */
    .doc-stat-card {
        background: #ffffff;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .doc-stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px -2px rgba(0, 0, 0, 0.08);
    }
    .stat-label {
        color: #64748B;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stat-value {
        color: #0F172A;
        font-size: 1.6rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .stat-sub {
        color: #10B981;
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 2px;
    }
    
    /* Red flag risk cards */
    .risk-card-high {
        background: #FEF2F2;
        border-left: 5px solid #EF4444;
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 12px;
    }
    .risk-card-medium {
        background: #FFFBEB;
        border-left: 5px solid #F59E0B;
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 12px;
    }
    .risk-title {
        font-weight: 700;
        font-size: 0.95rem;
        color: #991B1B;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .risk-context {
        font-size: 0.88rem;
        color: #374151;
        background: rgba(255, 255, 255, 0.7);
        padding: 8px 10px;
        border-radius: 6px;
        margin-top: 6px;
        font-style: italic;
    }
    .risk-recommendation {
        font-size: 0.86rem;
        color: #065F46;
        margin-top: 6px;
        font-weight: 600;
    }
    
    /* Citation box */
    .citation-box {
        background: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 12px;
        margin-top: 8px;
        font-size: 0.88rem;
    }
    </style>
    """, unsafe_allow_html=True)


def render_hero_banner():
    """Render top hero header."""
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">
            🧙‍♂️ DocuWiz AI
        </div>
        <div class="hero-subtitle">
            Enterprise Document Intelligence powered by <b>BART</b> summarization, <b>DistilBERT</b> sentiment & tone analysis, and source-grounded <b>RAG</b>.
        </div>
        <div class="hero-badges">
            <span class="hero-tag">⚡ BART Map-Reduce</span>
            <span class="hero-tag">🎭 DistilBERT Sentiment Arc</span>
            <span class="hero-tag">🔍 RAG + Page Provenance</span>
            <span class="hero-tag">⚖️ Legal Clause & Risk Audit</span>
            <span class="hero-tag">🎓 Academic Rigor Engine</span>
            <span class="hero-tag">🚀 60% Processing Speedup</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_metric_card(label: str, value: str, subtext: str = ""):
    """Render a styled metric card."""
    sub_html = f'<div class="stat-sub">{subtext}</div>' if subtext else ''
    st.markdown(f"""
    <div class="doc-stat-card">
        <div class="stat-label">{label}</div>
        <div class="stat-value">{value}</div>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)
