"""
DocuWiz AI - DistilBERT Sentiment & Tone Analysis Engine
Analyzes document polarity, confidence, tone progression, and risk indicators.
"""

import re
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

_SENTIMENT_CACHE: Dict[str, Any] = {}


def get_sentiment_pipeline(
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
    device: str = "cpu"
):
    """Lazy load and cache DistilBERT sentiment pipeline."""
    if model_name in ["fast-lexicon", "lexicon", "fallback", "fast"]:
        return None

    global _SENTIMENT_CACHE
    cache_key = f"{model_name}_{device}"
    if cache_key in _SENTIMENT_CACHE:
        return _SENTIMENT_CACHE[cache_key]

    try:
        from transformers import pipeline
        import torch

        device_id = 0 if device == "cuda" and torch.cuda.is_available() else -1
        classifier = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=device_id,
            truncation=True
        )
        _SENTIMENT_CACHE[cache_key] = classifier
        logger.info(f"Loaded DistilBERT sentiment model: {model_name}")
        return classifier
    except Exception as e:
        logger.warning(f"Could not load DistilBERT model {model_name}: {e}. Using rule-based fallback.")
        return None


def detect_tone(text: str) -> str:
    """
    Classify linguistic tone based on stylistic markers.
    Returns: 'Assertive', 'Cautious', 'Adversarial', 'Collaborative', or 'Formal Neutral'.
    """
    text_lower = text.lower()

    # Cautious / Speculative markers (common in academic limitations & risk disclaimers)
    cautious_markers = ["may", "might", "could", "perhaps", "possibly", "hypothetically", "suggest", "indicat", "subject to", "unforeseen"]
    # Adversarial / Penalty / Breach markers (common in legal contracts)
    adversarial_markers = ["breach", "indemnif", "liabilit", "damage", "penalt", "default", "terminat", "infring", "dispute", "litigat"]
    # Assertive / Definitive markers
    assertive_markers = ["shall", "must", "strictly", "guaranteed", "confirmed", "unconditionally", "conclusively", "demonstrat", "prove"]
    # Collaborative markers
    collaborative_markers = ["cooperat", "mutual", "agreed", "jointly", "partnership", "collaborat", "shared", "together"]

    cautious_count = sum(1 for m in cautious_markers if re.search(r'\b' + m, text_lower))
    adversarial_count = sum(1 for m in adversarial_markers if re.search(r'\b' + m, text_lower))
    assertive_count = sum(1 for m in assertive_markers if re.search(r'\b' + m, text_lower))
    collab_count = sum(1 for m in collaborative_markers if re.search(r'\b' + m, text_lower))

    scores = {
        "Cautious / Speculative": cautious_count,
        "Adversarial / Risk-Heavy": adversarial_count * 1.5,
        "Assertive / Binding": assertive_count,
        "Collaborative": collab_count * 1.2
    }

    best_tone, best_score = max(scores.items(), key=lambda x: x[1])
    if best_score >= 1.0:
        return best_tone
    return "Formal Neutral"


def rule_based_sentiment_fallback(text: str) -> Dict[str, Any]:
    """Lexicon-based sentiment scoring fallback."""
    pos_words = {
        "good", "great", "excellent", "superior", "benefit", "favorable", "compliant", "profit",
        "growth", "success", "innovative", "effective", "secure", "approved", "valid", "satisfied"
    }
    neg_words = {
        "bad", "fail", "failure", "breach", "penalty", "liability", "damage", "risk", "loss",
        "loss", "terminate", "dispute", "infringe", "violation", "adverse", "delay", "negligence", "default"
    }

    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    if not words:
        return {"label": "NEUTRAL", "score": 0.5, "polarity": 0.0}

    pos_count = sum(1 for w in words if w in pos_words)
    neg_count = sum(1 for w in words if w in neg_words)

    if pos_count > neg_count:
        polarity = min(1.0, (pos_count - neg_count) / max(5, len(words) * 0.1))
        return {"label": "POSITIVE", "score": round(0.5 + polarity * 0.5, 3), "polarity": round(polarity, 3)}
    elif neg_count > pos_count:
        polarity = max(-1.0, (neg_count - pos_count) / max(5, len(words) * 0.1) * -1)
        return {"label": "NEGATIVE", "score": round(0.5 + abs(polarity) * 0.5, 3), "polarity": round(polarity, 3)}
    else:
        return {"label": "NEUTRAL", "score": 0.5, "polarity": 0.0}


def analyze_sentiment(
    text: str,
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
    device: str = "cpu",
    granularity: str = "paragraph"
) -> Dict[str, Any]:
    """
    Analyze sentiment across document sections to generate overall metrics and timeline.

    Returns:
        Dict with overall label, confidence, timeline, tone breakdown, and top risk/positive segments.
    """
    if not text or len(text.strip()) < 10:
        return {
            "overall_label": "NEUTRAL",
            "overall_score": 0.5,
            "overall_polarity": 0.0,
            "dominant_tone": "Formal Neutral",
            "timeline": [],
            "distribution": {"positive": 0, "negative": 0, "neutral": 100},
            "tone_distribution": {}
        }

    # Split into sections/paragraphs for sentiment timeline
    if granularity == "paragraph":
        raw_sections = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 30]
        if len(raw_sections) < 3:
            # Fallback to sentences if document has few paragraphs
            raw_sections = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 30]
    else:
        raw_sections = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 25]

    # Limit sections to 50 max for performance
    if len(raw_sections) > 50:
        step = len(raw_sections) // 40
        raw_sections = raw_sections[::step][:40]

    pipeline = get_sentiment_pipeline(model_name=model_name, device=device)

    timeline: List[Dict[str, Any]] = []
    pos_count = 0
    neg_count = 0
    neut_count = 0
    tones_count: Dict[str, int] = {}

    for idx, sec in enumerate(raw_sections):
        # Truncate section snippet for model
        input_snippet = sec[:500]
        tone = detect_tone(sec)
        tones_count[tone] = tones_count.get(tone, 0) + 1

        if pipeline:
            try:
                res = pipeline(input_snippet)[0]
                lbl = res["label"].upper()
                scr = round(float(res["score"]), 3)
                polarity = scr if lbl == "POSITIVE" else -scr
            except Exception:
                fb = rule_based_sentiment_fallback(sec)
                lbl = fb["label"]
                scr = fb["score"]
                polarity = fb["polarity"]
        else:
            fb = rule_based_sentiment_fallback(sec)
            lbl = fb["label"]
            scr = fb["score"]
            polarity = fb["polarity"]

        if lbl == "POSITIVE":
            pos_count += 1
        elif lbl == "NEGATIVE":
            neg_count += 1
        else:
            neut_count += 1

        timeline.append({
            "section_index": idx + 1,
            "snippet": (sec[:140] + "...") if len(sec) > 140 else sec,
            "label": lbl,
            "score": scr,
            "polarity": polarity,
            "tone": tone
        })

    total_secs = max(1, len(timeline))
    pos_pct = round((pos_count / total_secs) * 100, 1)
    neg_pct = round((neg_count / total_secs) * 100, 1)
    neut_pct = round(100 - (pos_pct + neg_pct), 1)

    # Calculate overall polarity
    avg_polarity = round(sum(item["polarity"] for item in timeline) / total_secs, 3)
    if avg_polarity > 0.15:
        overall_label = "POSITIVE"
    elif avg_polarity < -0.15:
        overall_label = "NEGATIVE"
    else:
        overall_label = "NEUTRAL"

    dominant_tone = max(tones_count.items(), key=lambda x: x[1])[0] if tones_count else "Formal Neutral"

    # Identify most critical (negative/risk) and most positive segments
    sorted_by_polarity = sorted(timeline, key=lambda x: x["polarity"])
    critical_risk_passages = sorted_by_polarity[:3] if sorted_by_polarity and sorted_by_polarity[0]["polarity"] < 0 else []
    positive_highlights = sorted_by_polarity[-3:][::-1] if sorted_by_polarity and sorted_by_polarity[-1]["polarity"] > 0 else []

    return {
        "overall_label": overall_label,
        "overall_score": round(abs(avg_polarity), 3),
        "overall_polarity": avg_polarity,
        "dominant_tone": dominant_tone,
        "timeline": timeline,
        "distribution": {
            "positive": pos_pct,
            "negative": neg_pct,
            "neutral": max(0.0, neut_pct)
        },
        "tone_distribution": tones_count,
        "critical_risk_passages": critical_risk_passages,
        "positive_highlights": positive_highlights,
        "model_used": model_name if pipeline else "rule-based-lexicon"
    }
