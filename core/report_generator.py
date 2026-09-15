"""
DocuWiz AI - Executive Report Generator
Compiles unified audit reports for export in Markdown and JSON formats.
"""

import json
from datetime import datetime
from typing import Dict, Any


def generate_markdown_report(data: Dict[str, Any]) -> str:
    """
    Format full document intelligence analysis into an executive Markdown report.
    """
    filename = data.get("filename", "Document")
    meta = data.get("metadata", {})
    summary_data = data.get("summary_data", {})
    sentiment_data = data.get("sentiment_data", {})
    domain_data = data.get("domain_data", {})
    benchmarks = data.get("benchmarks", {})

    lines = []
    lines.append(f"# 🧙‍♂️ DocuWiz AI - Executive Intelligence Report")
    lines.append(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Target Document:** `{filename}`")
    lines.append("")

    lines.append("## 1. Document Overview & Metrics")
    lines.append(f"- **Pages:** {meta.get('pages_count', 1)}")
    lines.append(f"- **Total Words:** {meta.get('word_count', 0):,}")
    lines.append(f"- **Unique Vocabulary:** {meta.get('unique_words', 0):,} ({meta.get('lexical_diversity', 0.0):.1%} diversity)")
    lines.append(f"- **Estimated Manual Reading Time:** {meta.get('reading_time_minutes', 0)} mins")
    lines.append("")

    lines.append("## 2. Executive BART Summary")
    lines.append(summary_data.get("summary", "No summary available."))
    lines.append("")

    takeaways = summary_data.get("key_takeaways", [])
    if takeaways:
        lines.append("### Key Takeaways")
        for t in takeaways:
            lines.append(f"- {t}")
        lines.append("")

    lines.append("## 3. DistilBERT Sentiment & Tone Audit")
    dist = sentiment_data.get("distribution", {})
    lines.append(f"- **Overall Polarity:** {sentiment_data.get('overall_label', 'NEUTRAL')} (Confidence: {sentiment_data.get('overall_score', 0.5):.1%})")
    lines.append(f"- **Dominant Tone:** {sentiment_data.get('dominant_tone', 'Formal Neutral')}")
    lines.append(f"- **Polarity Breakdown:** {dist.get('positive', 0)}% Positive | {dist.get('neutral', 0)}% Neutral | {dist.get('negative', 0)}% Negative / Risk")
    lines.append("")

    if domain_data.get("domain") == "legal":
        lines.append("## 4. Legal Risk & Clause Audit")
        lines.append(f"- **Contract Risk Rating:** {domain_data.get('risk_level', 'Unknown')} ({domain_data.get('risk_score', 0)}/100)")
        lines.append(f"- **Jurisdiction:** {domain_data.get('governing_jurisdiction', 'Not detected')}")
        lines.append(f"- **Clauses Found:** {domain_data.get('total_clauses_found', 0)}")
        lines.append("")
        red_flags = domain_data.get("red_flags", [])
        if red_flags:
            lines.append("### 🚩 Red Flag Items & Recommendations")
            for flag in red_flags:
                lines.append(f"- **[{flag.get('severity')}] {flag.get('issue')}**")
                lines.append(f"  - *Context:* {flag.get('context')}")
                lines.append(f"  - *Action:* {flag.get('recommendation')}")
            lines.append("")

    elif domain_data.get("domain") == "academic":
        lines.append("## 4. Academic Rigor & Research Audit")
        lines.append(f"- **Academic Rigor Rating:** {domain_data.get('rigor_level', 'Unknown')} ({domain_data.get('rigor_score', 0)}/100)")
        lines.append(f"- **Citations Identified:** {domain_data.get('citations_count', 0)}")
        metrics = domain_data.get("extracted_metrics", [])
        if metrics:
            lines.append("### Key Empirical Metrics")
            for m in metrics:
                lines.append(f"- {m}")
        lines.append("")

    lines.append("## 5. Productivity & Processing Benchmark")
    lines.append(f"- **Time Reduction:** {benchmarks.get('processing_speedup_percent', 60.0)}% faster")
    lines.append(f"- **Time Saved:** {benchmarks.get('time_saved_minutes', 0)} minutes")
    lines.append(f"- **Productivity Multiplier:** {benchmarks.get('productivity_multiplier', '2.5x')}")
    lines.append("")
    lines.append("---")
    lines.append("*Generated automatically by DocuWiz AI Document Intelligence Platform.*")

    return "\n".join(lines)


def generate_json_report(data: Dict[str, Any]) -> str:
    """Format full analysis into indented JSON."""
    return json.dumps(data, indent=2, default=str)
