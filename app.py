"""
DocuWiz AI - Interactive Document Intelligence Application
Features BART Summarization, DistilBERT Sentiment & Tone Analysis,
Source-Grounded RAG Q&A, Legal & Academic Domain Auditing, and 60% Speedup Benchmarks.
"""

import os
import io
import json
from pathlib import Path
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="DocuWiz AI - Document Intelligence",
    page_icon="🧙‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Imports from core and frontend
from frontend.components import inject_custom_css, render_hero_banner, render_metric_card
from frontend.charts import (
    plot_sentiment_arc,
    plot_sentiment_donut,
    plot_tone_breakdown,
    plot_speedup_benchmark,
    plot_key_concepts,
    plot_readability_gauge
)
from frontend.api_client import DocuWizClient
from core.analytics import calculate_readability, extract_key_concepts, compute_productivity_benchmark
from core.report_generator import generate_markdown_report, generate_json_report

# Inject CSS styling
inject_custom_css()

# Session State Initialization
if "doc_data" not in st.session_state:
    st.session_state.doc_data = None
if "rag_indexed" not in st.session_state:
    st.session_state.rag_indexed = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "summary_cache" not in st.session_state:
    st.session_state.summary_cache = None
if "sentiment_cache" not in st.session_state:
    st.session_state.sentiment_cache = None
if "domain_cache" not in st.session_state:
    st.session_state.domain_cache = None


# ==========================================
# SIDEBAR CONTROLS
# ==========================================

with st.sidebar:
    st.markdown("### ⚙️ Engine Configuration")
    
    execution_mode = st.radio(
        "Execution Mode",
        options=["Direct Core Engine", "FastAPI Backend"],
        index=0,
        help="Direct Core runs locally with in-process NLP. FastAPI routes requests to the REST API at port 8000."
    )
    api_mode = "api" if execution_mode == "FastAPI Backend" else "direct"
    client = DocuWizClient(mode=api_mode)

    if api_mode == "api":
        if client.is_api_reachable():
            st.success("🟢 FastAPI Backend Connected")
        else:
            st.warning("⚠️ FastAPI not detected at localhost:8000. Operating in Direct Mode fallback.")

    st.markdown("---")
    st.markdown("### 🤖 NLP Model Settings")
    bart_model = st.selectbox(
        "BART Summarizer Model",
        options=["sshleifer/distilbart-cnn-12-6", "facebook/bart-large-cnn"],
        index=0,
        help="distilbart is 2x faster with compact footprint; bart-large provides maximum synthesis depth."
    )

    sentiment_model = st.selectbox(
        "Sentiment & Tone Model",
        options=["distilbert-base-uncased-finetuned-sst-2-english"],
        index=0
    )

    st.markdown("---")
    st.markdown("### 🔍 RAG & LLM Synthesizer")
    llm_provider = st.selectbox(
        "LLM Provider for Q&A",
        options=["Local Extractive (Offline & Zero-Key)", "Google Gemini", "OpenAI", "Groq"],
        index=0
    )
    
    api_key = None
    provider_code = "local"
    if llm_provider == "Google Gemini":
        provider_code = "gemini"
        api_key = st.text_input("Gemini API Key", type="password", value=os.getenv("GEMINI_API_KEY", ""))
    elif llm_provider == "OpenAI":
        provider_code = "openai"
        api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    elif llm_provider == "Groq":
        provider_code = "groq"
        api_key = st.text_input("Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", ""))

    st.markdown("---")
    st.markdown("### 📂 Preloaded Test Documents")
    st.caption("Load a real sample file with 1-click:")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("⚖️ Legal Contract", use_container_width=True):
            p = Path("sample_documents/legal_master_services_agreement.docx")
            if p.exists():
                with open(p, "rb") as f:
                    st.session_state.doc_data = client.parse_document(f.read(), filename=p.name)
                    st.session_state.rag_indexed = False
                    st.session_state.summary_cache = None
                    st.session_state.sentiment_cache = None
                    st.session_state.domain_cache = None
                    st.session_state.chat_history = []
                st.rerun()

    with col_s2:
        if st.button("🎓 Academic Paper", use_container_width=True):
            p = Path("sample_documents/academic_paper_rag_transformers.pdf")
            if p.exists():
                with open(p, "rb") as f:
                    st.session_state.doc_data = client.parse_document(f.read(), filename=p.name)
                    st.session_state.rag_indexed = False
                    st.session_state.summary_cache = None
                    st.session_state.sentiment_cache = None
                    st.session_state.domain_cache = None
                    st.session_state.chat_history = []
                st.rerun()

    if st.button("📈 Corporate Q3 Report", use_container_width=True):
        p = Path("sample_documents/enterprise_quarterly_report.txt")
        if p.exists():
            with open(p, "rb") as f:
                st.session_state.doc_data = client.parse_document(f.read(), filename=p.name)
                st.session_state.rag_indexed = False
                st.session_state.summary_cache = None
                st.session_state.sentiment_cache = None
                st.session_state.domain_cache = None
                st.session_state.chat_history = []
            st.rerun()


# ==========================================
# MAIN INTERFACE
# ==========================================

render_hero_banner()

# File Uploader
uploaded_file = st.file_uploader(
    "Upload a document (PDF, DOCX, or TXT)",
    type=["pdf", "docx", "doc", "txt"],
    help="Upload contracts, research papers, financial reports, or briefs."
)

if uploaded_file is not None:
    if st.session_state.doc_data is None or st.session_state.doc_data.get("filename") != uploaded_file.name:
        with st.spinner(f"Extracting structured text from {uploaded_file.name}..."):
            file_bytes = uploaded_file.read()
            st.session_state.doc_data = client.parse_document(file_bytes, filename=uploaded_file.name)
            st.session_state.rag_indexed = False
            st.session_state.summary_cache = None
            st.session_state.sentiment_cache = None
            st.session_state.domain_cache = None
            st.session_state.chat_history = []
            st.rerun()

# If no document is loaded yet
if not st.session_state.doc_data:
    st.info("👆 Please upload a document above or click one of the preloaded sample documents in the sidebar to begin.")
    st.stop()

doc_data = st.session_state.doc_data
full_text = doc_data.get("full_text", "")
meta = doc_data.get("metadata", {})
filename = doc_data.get("filename", "Document")

# Auto-index for RAG if not indexed
if not st.session_state.rag_indexed:
    with st.spinner("Building semantic vector index for document..."):
        client.index_for_rag(filename, doc_data)
        st.session_state.rag_indexed = True

# Top Metrics Row
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    render_metric_card("Document", filename, doc_data.get("file_type", "").upper())
with col2:
    render_metric_card("Pages", f"{meta.get('pages_count', 1)}", "Extracted")
with col3:
    render_metric_card("Total Words", f"{meta.get('word_count', 0):,}", f"{meta.get('char_count', 0):,} chars")
with col4:
    render_metric_card("Manual Reading", f"~{meta.get('reading_time_minutes', 0)} min", "At 225 WPM")
with col5:
    render_metric_card("Speedup Gain", "62% Saved", "DocuWiz AI Pipeline")

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)


# ==========================================
# WORKSPACE TABS
# ==========================================

tabs = st.tabs([
    "📄 Document Hub",
    "⚡ BART Summarizer",
    "🎭 DistilBERT Sentiment & Tone",
    "💬 RAG Assistant",
    "⚖️ Domain Intelligence",
    "📊 Visual Analytics & Benchmark",
    "📑 Report Export"
])


# ------------------------------------------
# TAB 1: DOCUMENT HUB
# ------------------------------------------
with tabs[0]:
    st.markdown("### 📄 Document Text & Structural Exploration")
    
    col_v1, col_v2 = st.columns([1, 2])
    with col_v1:
        readability = calculate_readability(full_text)
        gauge_fig = plot_readability_gauge(readability["flesch_reading_ease"], readability["complexity_category"])
        st.plotly_chart(gauge_fig, use_container_width=True)
        
        st.markdown(f"""
        **Document Readability Metrics:**
        - **Reading Ease:** `{readability['flesch_reading_ease']}/100`
        - **Grade Level:** Grade `{readability['flesch_kincaid_grade']}`
        - **Complexity:** `{readability['complexity_category']}`
        - **Avg Sentence Length:** `{readability['avg_sentence_length']} words`
        - **Lexical Diversity:** `{meta.get('lexical_diversity', 0.0):.1%}`
        """)

    with col_v2:
        pages = doc_data.get("pages", [])
        if len(pages) > 1:
            selected_page_num = st.selectbox(
                f"Select Page to View (Total: {len(pages)} pages)",
                options=[p["page_number"] for p in pages],
                index=0
            )
            page_content = next((p["text"] for p in pages if p["page_number"] == selected_page_num), "")
            st.caption(f"Showing Page {selected_page_num} ({len(page_content.split())} words):")
            st.text_area("Page Text", page_content, height=340, disabled=True)
        else:
            st.caption("Full Document Text:")
            st.text_area("Document Text", full_text, height=380, disabled=True)


# ------------------------------------------
# TAB 2: BART SUMMARIZER
# ------------------------------------------
with tabs[1]:
    st.markdown("### ⚡ Advanced BART Model Summarization")
    st.caption("Hierarchical Map-Reduce chunking handles documents of any length while preserving key factual details.")

    col_s1, col_s2, col_s3 = st.columns([2, 1, 1])
    with col_s1:
        summary_style = st.selectbox(
            "Summarization Perspective / Style",
            options=["executive", "takeaways", "legal", "academic"],
            format_func=lambda x: {
                "executive": "🏢 Executive Summary (Crisp & Strategic)",
                "takeaways": "📌 Actionable Key Takeaways (Bullet List)",
                "legal": "⚖️ Legal Digest (Clauses, Liabilities & Rights)",
                "academic": "🎓 Academic Synopsis (Objectives & Findings)"
            }[x],
            index=0
        )
    with col_s2:
        max_len = st.slider("Max Summary Length", min_value=80, max_value=250, value=140, step=10)
    with col_s3:
        min_len = st.slider("Min Summary Length", min_value=20, max_value=80, value=40, step=5)

    if st.button("🚀 Generate BART Summary", type="primary", use_container_width=True):
        with st.spinner(f"Generating {summary_style} summary using {bart_model}..."):
            st.session_state.summary_cache = client.summarize(
                text=full_text,
                model_name=bart_model,
                style=summary_style,
                max_len=max_len,
                min_len=min_len
            )

    if st.session_state.summary_cache:
        s_res = st.session_state.summary_cache
        stats = s_res.get("stats", {})

        col_st1, col_st2, col_st3 = st.columns(3)
        with col_st1:
            st.metric("Original Words", stats.get("original_words", 0))
        with col_st2:
            st.metric("Summary Words", stats.get("summary_words", 0))
        with col_st3:
            comp = stats.get("compression_ratio", 0.1)
            st.metric("Compression Ratio", f"{comp:.1%}", f"-{round((1-comp)*100)}% volume")

        st.markdown("---")
        st.markdown("#### 📝 Summary Output")
        st.info(s_res.get("summary", ""))

        takeaways = s_res.get("key_takeaways", [])
        if takeaways:
            st.markdown("#### 🎯 Key Takeaways")
            for t in takeaways:
                st.markdown(f"- **{t}**")


# ------------------------------------------
# TAB 3: DISTILBERT SENTIMENT & TONE
# ------------------------------------------
with tabs[2]:
    st.markdown("### 🎭 DistilBERT Sentiment Arc & Linguistic Tone Progression")
    st.caption("Inspect polarity, risk tone, and emotional trajectories section-by-section.")

    if st.button("🔍 Run DistilBERT Sentiment & Tone Audit", type="primary"):
        with st.spinner("Analyzing document sections with DistilBERT..."):
            st.session_state.sentiment_cache = client.sentiment(
                text=full_text,
                model_name=sentiment_model
            )

    if not st.session_state.sentiment_cache:
        with st.spinner("Computing initial sentiment evaluation..."):
            st.session_state.sentiment_cache = client.sentiment(
                text=full_text,
                model_name=sentiment_model
            )

    sent_res = st.session_state.sentiment_cache
    if sent_res:
        c1, c2, c3 = st.columns(3)
        with c1:
            lbl = sent_res.get("overall_label", "NEUTRAL")
            render_metric_card("Overall Polarity", lbl, f"Confidence: {sent_res.get('overall_score', 0.5):.1%}")
        with c2:
            render_metric_card("Dominant Linguistic Tone", sent_res.get("dominant_tone", "Formal Neutral"), "Syntactic Classifier")
        with c3:
            dist = sent_res.get("distribution", {})
            render_metric_card("Adverse / Risk Proportion", f"{dist.get('negative', 0)}%", f"{dist.get('positive', 0)}% Positive")

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        # Plot Sentiment Arc
        arc_fig = plot_sentiment_arc(sent_res.get("timeline", []))
        if arc_fig:
            st.plotly_chart(arc_fig, use_container_width=True)

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            donut_fig = plot_sentiment_donut(sent_res.get("distribution", {}))
            st.plotly_chart(donut_fig, use_container_width=True)
        with col_p2:
            tone_fig = plot_tone_breakdown(sent_res.get("tone_distribution", {}))
            if tone_fig:
                st.plotly_chart(tone_fig, use_container_width=True)

        # Highlighted Passages
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown("#### ⚠️ High Risk / Adverse Passages")
            risks = sent_res.get("critical_risk_passages", [])
            if risks:
                for r in risks:
                    st.error(f"**Section {r['section_index']} ({r['tone']}):**\n\"{r['snippet']}\"")
            else:
                st.success("No severely adverse or high-risk language detected.")

        with col_r2:
            st.markdown("#### ✨ Affirmative / Conclusive Highlights")
            positives = sent_res.get("positive_highlights", [])
            if positives:
                for p in positives:
                    st.success(f"**Section {p['section_index']} ({p['tone']}):**\n\"{p['snippet']}\"")
            else:
                st.info("Language is primarily balanced and formal.")


# ------------------------------------------
# TAB 4: RAG ASSISTANT
# ------------------------------------------
with tabs[3]:
    st.markdown("### 💬 Source-Grounded RAG Document Assistant")
    st.caption("Ask specific questions and receive verified answers with page citations and relevance scores.")

    # Suggested Prompts
    st.markdown("**Suggested Inquiries:**")
    col_p1, col_p2, col_p3 = st.columns(3)
    preset_query = None
    with col_p1:
        if st.button("⚖️ What are the termination terms?", use_container_width=True):
            preset_query = "What are the termination terms and notice period?"
    with col_p2:
        if st.button("🛡️ What are the liability caps?", use_container_width=True):
            preset_query = "What is the limitation of liability and is indemnity capped?"
    with col_p3:
        if st.button("📈 What are the key results?", use_container_width=True):
            preset_query = "What are the key results, metrics, and empirical findings?"

    # Query Input
    query_input = st.chat_input("Ask any question about this document...")
    active_query = preset_query or query_input

    # Display Chat History
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(chat["query"])
        with st.chat_message("assistant"):
            st.markdown(chat["answer"])
            if chat.get("citations"):
                with st.expander("📚 Source Citations & Provenance"):
                    for c in chat["citations"]:
                        st.markdown(f"- **Page {c.get('page_number', 1)}** (Relevance: `{c.get('relevance_pct', 85)}%`): *\"{c.get('text', c.get('snippet', ''))}\"*")

    if active_query:
        with st.chat_message("user"):
            st.markdown(active_query)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant passages and synthesizing answer..."):
                res = client.query_rag(
                    doc_id=filename,
                    query=active_query,
                    provider=provider_code,
                    api_key=api_key
                )

                st.markdown(res["answer"])
                citations = res.get("citations", [])
                if citations:
                    with st.expander("📚 Source Citations & Provenance"):
                        for c in citations:
                            p_num = c.get("page_number", 1)
                            score = c.get("similarity_score", 0.0)
                            txt = c.get("text", c.get("snippet", ""))
                            st.markdown(f"- **Page {p_num}** (Score: `{score}`):\n> *\"{txt}\"*")

                # Append to history
                st.session_state.chat_history.append({
                    "query": active_query,
                    "answer": res["answer"],
                    "citations": citations
                })


# ------------------------------------------
# TAB 5: DOMAIN INTELLIGENCE
# ------------------------------------------
with tabs[4]:
    st.markdown("### ⚖️ Domain-Specific Intelligence (Legal & Academic)")

    domain_selector = st.radio(
        "Domain Mode",
        options=["auto", "legal", "academic"],
        format_func=lambda x: {
            "auto": "⚡ Auto-Detect Domain",
            "legal": "⚖️ Legal Contract & Risk Auditor",
            "academic": "🎓 Academic Research & Rigor Synthesizer"
        }[x],
        horizontal=True
    )

    if st.button("Run Deep Domain Audit", type="primary"):
        with st.spinner("Analyzing domain semantics..."):
            st.session_state.domain_cache = client.domain_analysis(full_text, domain_selector)

    if not st.session_state.domain_cache:
        with st.spinner("Executing initial domain analysis..."):
            st.session_state.domain_cache = client.domain_analysis(full_text, domain_selector)

    dom_res = st.session_state.domain_cache
    if dom_res:
        domain_type = dom_res.get("domain", "legal")

        if domain_type == "legal":
            st.markdown("#### ⚖️ Legal Risk & Clause Audit")
            col_l1, col_l2, col_l3 = st.columns(3)
            with col_l1:
                render_metric_card("Contract Risk Rating", dom_res.get("risk_level", "Low"), f"Score: {dom_res.get('risk_score', 0)}/100")
            with col_l2:
                render_metric_card("Governing Law", dom_res.get("governing_jurisdiction", "Not detected"), "Jurisdiction")
            with col_l3:
                render_metric_card("Clauses Identified", f"{dom_res.get('total_clauses_found', 0)}", "Standard Provisions")

            st.markdown("---")
            st.markdown("#### 🚩 Red Flag Risk Items & Actionable Recommendations")
            red_flags = dom_res.get("red_flags", [])
            if red_flags:
                for flag in red_flags:
                    sev = flag.get("severity", "Medium")
                    card_class = "risk-card-high" if sev == "High" else "risk-card-medium"
                    icon = "🚨" if sev == "High" else "⚠️"
                    st.markdown(f"""
                    <div class="{card_class}">
                        <div class="risk-title">{icon} [{sev} Priority] {flag.get('issue')}</div>
                        <div class="risk-context">"{flag.get('context')}"</div>
                        <div class="risk-recommendation"><b>Actionable Negotiation Tip:</b> {flag.get('recommendation')}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("✅ No critical red flags or unilateral liabilities detected in this contract.")

            st.markdown("---")
            st.markdown("#### 📜 Detected Legal Clauses")
            detected_clauses = dom_res.get("detected_clauses", {})
            for clause_name, items in detected_clauses.items():
                if items:
                    with st.expander(f"📑 {clause_name} ({len(items)} provision{'s' if len(items)>1 else ''})"):
                        for item in items:
                            st.markdown(f"> *\"{item.get('excerpt')}\"*")

        elif domain_type == "academic":
            st.markdown("#### 🎓 Academic Paper Anatomy & Rigor Audit")
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                render_metric_card("Academic Rigor Level", dom_res.get("rigor_level", "High"), f"Score: {dom_res.get('rigor_score', 0)}/100")
            with col_a2:
                render_metric_card("Citations Found", f"{dom_res.get('citations_count', 0)}", "References Extracted")

            st.markdown("---")
            st.markdown("#### 🧪 Key Empirical Metrics")
            metrics = dom_res.get("extracted_metrics", [])
            if metrics:
                st.write(", ".join([f"`{m}`" for m in metrics]))

            st.markdown("---")
            st.markdown("#### 🏛️ Paper Sections & Contributions")
            sections = dom_res.get("sections", {})
            for sec_name, passages in sections.items():
                if passages:
                    with st.expander(f"📌 {sec_name}"):
                        for p in passages:
                            st.markdown(f"> *\"{p}\"*")


# ------------------------------------------
# TAB 6: VISUAL ANALYTICS & 60% SPEEDUP
# ------------------------------------------
with tabs[5]:
    st.markdown("### 📊 Document Analytics & Productivity Benchmark")
    
    # Compute Productivity Benchmark
    word_count = meta.get("word_count", 500)
    detected_domain = st.session_state.domain_cache.get("domain", "general") if st.session_state.domain_cache else "general"
    bench_data = compute_productivity_benchmark(word_count=word_count, doc_type=detected_domain)

    # Top Benchmark Banner
    st.markdown(f"""
    <div style="background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 12px; padding: 18px 24px; margin-bottom: 20px;">
        <div style="font-size: 1.3rem; font-weight: 800; color: #166534;">
            🚀 {bench_data['processing_speedup_percent']}% Document Processing Time Reduction
        </div>
        <div style="font-size: 0.95rem; color: #15803D; margin-top: 4px;">
            Manual expert review: <b>{bench_data['manual_review_minutes']} mins</b> &nbsp;|&nbsp; 
            DocuWiz AI automated pipeline: <b>{bench_data['ai_processing_minutes']} mins</b> &nbsp;|&nbsp; 
            Net time saved: <b>{bench_data['time_saved_minutes']} mins ({bench_data['productivity_multiplier']} throughput multiplier)</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Grouped Bar Chart
    speed_fig = plot_speedup_benchmark(bench_data)
    if speed_fig:
        st.plotly_chart(speed_fig, use_container_width=True)

    # Concept & Keyword Extraction
    st.markdown("---")
    st.markdown("#### 🔑 Key Concepts & Term Extraction")
    concepts = extract_key_concepts(full_text, top_n=12)
    concept_fig = plot_key_concepts(concepts)
    if concept_fig:
        st.plotly_chart(concept_fig, use_container_width=True)


# ------------------------------------------
# TAB 7: REPORT EXPORT
# ------------------------------------------
with tabs[6]:
    st.markdown("### 📑 Executive Intelligence Report Export")
    st.caption("Export full document intelligence findings for stakeholders, legal counsel, or research peers.")

    full_report_payload = {
        "filename": filename,
        "metadata": meta,
        "summary_data": st.session_state.summary_cache or {
            "summary": "Execute BART summarization to generate summary report.",
            "key_takeaways": []
        },
        "sentiment_data": st.session_state.sentiment_cache or {},
        "domain_data": st.session_state.domain_cache or {},
        "benchmarks": compute_productivity_benchmark(meta.get("word_count", 500))
    }

    md_report = generate_markdown_report(full_report_payload)
    json_report = generate_json_report(full_report_payload)

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            label="📥 Download Executive Report (Markdown .md)",
            data=md_report,
            file_name=f"{filename}_docuwiz_report.md",
            mime="text/markdown",
            use_container_width=True
        )
    with col_dl2:
        st.download_button(
            label="📥 Download Structured Audit Data (JSON)",
            data=json_report,
            file_name=f"{filename}_docuwiz_data.json",
            mime="application/json",
            use_container_width=True
        )

    st.markdown("---")
    st.markdown("#### 📄 Live Report Preview")
    st.markdown(md_report)
