"""
DocuWiz AI - Interactive Plotly Visualizations
Provides sentiment timelines, tone distribution, domain concepts, and productivity benchmarks.
"""

import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List


def plot_sentiment_arc(timeline: List[Dict[str, Any]]):
    """
    Plot emotional arc / sentiment progression across document sections.
    """
    if not timeline:
        return None

    indices = [item["section_index"] for item in timeline]
    polarities = [item["polarity"] for item in timeline]
    hover_texts = [f"Section {item['section_index']}<br>Tone: {item['tone']}<br>Excerpt: {item['snippet']}" for item in timeline]
    labels = [item["label"] for item in timeline]

    # Color code positive (green), neutral (blue/grey), negative (red)
    colors = ["#10B981" if p > 0.1 else ("#EF4444" if p < -0.1 else "#64748B") for p in polarities]

    fig = go.Figure()

    # Base line
    fig.add_trace(go.Scatter(
        x=indices,
        y=polarities,
        mode="lines+markers",
        line=dict(color="#6366F1", width=3, shape="spline"),
        marker=dict(size=8, color=colors),
        hovertext=hover_texts,
        hoverinfo="text",
        name="Sentiment Polarity"
    ))

    # Reference zero line
    fig.add_hline(y=0, line_dash="dash", line_color="#94A3B8", annotation_text="Neutral", annotation_position="bottom right")

    fig.update_layout(
        title="<b>Document Sentiment Progression Arc (DistilBERT)</b>",
        xaxis_title="Document Section Sequence",
        yaxis_title="Polarity Score (-1.0 Negative to +1.0 Positive)",
        yaxis=dict(range=[-1.05, 1.05]),
        margin=dict(l=40, r=40, t=50, b=40),
        height=350,
        template="plotly_white"
    )

    return fig


def plot_sentiment_donut(distribution: Dict[str, float]):
    """
    Donut chart of document polarity breakdown.
    """
    labels = ["Positive", "Neutral", "Negative / Risk"]
    values = [
        distribution.get("positive", 0),
        distribution.get("neutral", 0),
        distribution.get("negative", 0)
    ]
    colors = ["#10B981", "#94A3B8", "#EF4444"]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors),
        textinfo="label+percent",
        hoverinfo="label+value"
    )])

    fig.update_layout(
        title="<b>Polarity Distribution</b>",
        margin=dict(l=20, r=20, t=50, b=20),
        height=320,
        template="plotly_white",
        showlegend=False
    )
    return fig


def plot_tone_breakdown(tone_counts: Dict[str, int]):
    """
    Horizontal bar chart of linguistic tone markers.
    """
    if not tone_counts:
        return None

    tones = list(tone_counts.keys())
    counts = list(tone_counts.values())

    fig = go.Figure(go.Bar(
        x=counts,
        y=tones,
        orientation="h",
        marker=dict(color="#4F46E5"),
        text=counts,
        textposition="auto"
    ))

    fig.update_layout(
        title="<b>Linguistic Tone Breakdown</b>",
        xaxis_title="Detected Frequency",
        yaxis_title="",
        margin=dict(l=20, r=20, t=50, b=20),
        height=320,
        template="plotly_white"
    )
    return fig


def plot_speedup_benchmark(benchmark_data: Dict[str, Any]):
    """
    Comparison chart demonstrating 60% document processing speedup.
    """
    breakdown = benchmark_data.get("benchmark_breakdown", [])
    if not breakdown:
        return None

    tasks = [item["task"] for item in breakdown]
    manual_times = [item["manual_mins"] for item in breakdown]
    docuwiz_times = [item["docuwiz_mins"] for item in breakdown]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Manual Review (mins)",
        x=tasks,
        y=manual_times,
        marker_color="#F87171",
        text=[f"{m}m" for m in manual_times],
        textposition="auto"
    ))
    fig.add_trace(go.Bar(
        name="DocuWiz AI (mins)",
        x=tasks,
        y=docuwiz_times,
        marker_color="#34D399",
        text=[f"{d}m" for d in docuwiz_times],
        textposition="auto"
    ))

    fig.update_layout(
        title=f"<b>Productivity Benchmark: {benchmark_data.get('processing_speedup_percent', 60)}% Time Reduction ({benchmark_data.get('productivity_multiplier', '2.5x')} Faster)</b>",
        yaxis_title="Time Spent (Minutes)",
        barmode="group",
        margin=dict(l=30, r=30, t=60, b=40),
        height=380,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def plot_key_concepts(concepts: List[Dict[str, Any]]):
    """
    Bar chart showing extracted domain keywords and multi-word terms.
    """
    if not concepts:
        return None

    terms = [item["term"] for item in concepts[:10]][::-1]
    frequencies = [item["frequency"] for item in concepts[:10]][::-1]
    types = [item["type"] for item in concepts[:10]][::-1]
    colors = ["#6366F1" if t == "Phrase" else "#3B82F6" for t in types]

    fig = go.Figure(go.Bar(
        x=frequencies,
        y=terms,
        orientation="h",
        marker_color=colors,
        text=frequencies,
        textposition="auto"
    ))

    fig.update_layout(
        title="<b>Top Domain Concepts & Phrases</b>",
        xaxis_title="Term Occurrences",
        margin=dict(l=20, r=20, t=50, b=20),
        height=380,
        template="plotly_white"
    )
    return fig


def plot_readability_gauge(fre_score: float, category: str):
    """
    Gauge indicator for Flesch Reading Ease score.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=fre_score,
        title={'text': f"<b>Readability Ease</b><br><span style='font-size:0.8em;color:gray'>{category}</span>"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#4338CA"},
            'steps': [
                {'range': [0, 30], 'color': "#FEE2E2"},
                {'range': [30, 60], 'color': "#FEF3C7"},
                {'range': [60, 100], 'color': "#DCFCE7"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 3},
                'thickness': 0.75,
                'value': fre_score
            }
        }
    ))

    fig.update_layout(
        margin=dict(l=30, r=30, t=50, b=30),
        height=260,
        template="plotly_white"
    )
    return fig
