"""
DocuWiz AI - Domain Intelligence Engine (Legal & Academic Analysis)
Provides deep domain-specific analysis for legal agreements and academic research papers.
"""

import re
from typing import Dict, Any, List, Optional


# ==========================================
# 1. LEGAL INTELLIGENCE
# ==========================================

LEGAL_CLAUSE_PATTERNS = {
    "Indemnification": [
        r"indemnif(?:y|ication|ied)",
        r"hold harmless",
        r"defend.*against.*claims"
    ],
    "Limitation of Liability": [
        r"limitation of liability",
        r"in no event shall.*liable",
        r"aggregate liability",
        r"consequential.*punitive.*damages",
        r"indirect.*special.*damages"
    ],
    "Termination": [
        r"termination for convenience",
        r"termination for cause",
        r"notice of termination",
        r"either party may terminate",
        r"survive.*termination"
    ],
    "Confidentiality": [
        r"confidential information",
        r"non-disclosure",
        r"proprietary information",
        r"standard of care.*confidential"
    ],
    "Governing Law & Jurisdiction": [
        r"governed by the laws of",
        r"exclusive jurisdiction",
        r"dispute resolution",
        r"arbitration.*rules"
    ],
    "Intellectual Property": [
        r"intellectual property rights",
        r"work made for hire",
        r"assignment of inventions",
        r"ownership of deliverables",
        r"patent.*copyright.*trademark"
    ],
    "Warranty & Disclaimers": [
        r"as is.*without warranty",
        r"disclaimer of warranties",
        r"express or implied warranties",
        r"merchantability.*fitness for a particular purpose"
    ],
    "Force Majeure": [
        r"force majeure",
        r"acts of god",
        r"beyond reasonable control"
    ]
}

RED_FLAG_PATTERNS = [
    {
        "issue": "Uncapped or Missing Liability Ceiling",
        "severity": "High",
        "regex": r"(unlimited liability|shall not be limited|no limitation on liability)",
        "recommendation": "Negotiate an explicit monetary liability cap (e.g., 1x or 2x 12-month fees paid)."
    },
    {
        "issue": "Unilateral Indemnification",
        "severity": "High",
        "regex": r"(customer shall indemnify|client agrees to indemnify and hold harmless)(?!.*supplier shall indemnify)",
        "recommendation": "Request mutual indemnification covering intellectual property infringement and gross negligence."
    },
    {
        "issue": "Short Termination Notice (< 15 Days)",
        "severity": "Medium",
        "regex": r"terminate.*(?:upon|within)\s*(?:[1-9]|1[0-4])\s*days",
        "recommendation": "Extend notice period to standard 30 or 60 calendar days for adequate operational transition."
    },
    {
        "issue": "Sole Discretion Decision Rights",
        "severity": "Medium",
        "regex": r"in its sole (?:and absolute )?discretion",
        "recommendation": "Amend standard to 'in its reasonable discretion' or 'commercially reasonable efforts'."
    },
    {
        "issue": "Auto-Renewal Lock-In Trap",
        "severity": "Medium",
        "regex": r"(automatically renew|automatic renewal)(?:.*(?:unless written notice|at least \d+ days))?",
        "recommendation": "Ensure opt-out reminder notice is required 60 days before expiration."
    },
    {
        "issue": "Non-Compete or Exclusivity Restriction",
        "severity": "Low",
        "regex": r"(exclusive provider|shall not compete|covenant not to compete|non-solicitation)",
        "recommendation": "Verify geographic scope and duration do not restrict core business expansion."
    }
]


def analyze_legal_document(text: str) -> Dict[str, Any]:
    """
    Perform deep legal audit: clause detection, red-flag risk evaluation, and key obligations.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 30]

    detected_clauses: Dict[str, List[Dict[str, Any]]] = {}
    for clause_name, patterns in LEGAL_CLAUSE_PATTERNS.items():
        detected_clauses[clause_name] = []
        for p in paragraphs:
            for pat in patterns:
                if re.search(pat, p, re.IGNORECASE):
                    detected_clauses[clause_name].append({
                        "excerpt": (p[:300] + "...") if len(p) > 300 else p,
                        "match_pattern": pat
                    })
                    break

    # Evaluate Red Flags
    red_flags: List[Dict[str, Any]] = []
    risk_points = 0

    for flag in RED_FLAG_PATTERNS:
        matches = re.finditer(flag["regex"], text, re.IGNORECASE)
        found = False
        for m in matches:
            found = True
            # Get surrounding context
            start = max(0, m.start() - 80)
            end = min(len(text), m.end() + 150)
            context = text[start:end].replace("\n", " ").strip()

            severity = flag["severity"]
            weight = 30 if severity == "High" else (15 if severity == "Medium" else 5)
            risk_points += weight

            red_flags.append({
                "issue": flag["issue"],
                "severity": severity,
                "context": f"...{context}...",
                "recommendation": flag["recommendation"]
            })
            break  # Record once per rule

    # Normalize Contract Risk Score (0-100)
    risk_score = min(100, max(10, risk_points + (20 if not detected_clauses.get("Limitation of Liability") else 0)))

    if risk_score >= 70:
        risk_level = "High Risk"
    elif risk_score >= 40:
        risk_level = "Moderate Risk"
    else:
        risk_level = "Low Risk / Standard"

    # Detect Governing Law / Jurisdiction
    gov_law_match = re.search(r"governed by.*laws of\s+([A-Z][a-zA-Z\s,]+?)(?:\.|\;|\n)", text, re.IGNORECASE)
    governing_jurisdiction = gov_law_match.group(1).strip() if gov_law_match else "Not explicitly identified"

    # Count identified clauses
    clauses_summary = {k: len(v) for k, v in detected_clauses.items()}

    return {
        "domain": "legal",
        "risk_score": risk_score,
        "risk_level": risk_level,
        "governing_jurisdiction": governing_jurisdiction,
        "clauses_summary": clauses_summary,
        "detected_clauses": detected_clauses,
        "red_flags": red_flags,
        "total_clauses_found": sum(clauses_summary.values())
    }


# ==========================================
# 2. ACADEMIC INTELLIGENCE
# ==========================================

ACADEMIC_SECTION_PATTERNS = {
    "Abstract & Objectives": [r"abstract", r"objective", r"we propose", r"in this paper, we", r"we introduce"],
    "Methodology & Architecture": [r"methodology", r"proposed approach", r"architecture", r"framework", r"formulation", r"model pipeline"],
    "Experimental Results": [r"experimental results", r"empirical evaluation", r"benchmarks", r"baseline", r"ablation study", r"performance metrics"],
    "Limitations & Threats": [r"limitations", r"threats to validity", r"future work", r"potential drawbacks", r"unresolved challenges"]
}


def analyze_academic_document(text: str) -> Dict[str, Any]:
    """
    Perform deep academic paper synthesis: hypotheses, methodologies, benchmarks, and limitations.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 30]

    sections: Dict[str, List[str]] = {k: [] for k in ACADEMIC_SECTION_PATTERNS}

    for p in paragraphs:
        for sec_name, pats in ACADEMIC_SECTION_PATTERNS.items():
            for pat in pats:
                if re.search(r'\b' + pat + r'\b', p, re.IGNORECASE):
                    sections[sec_name].append(p[:350] + "..." if len(p) > 350 else p)
                    break

    # Extract Citations & References
    citations = re.findall(r'\[(?:\d+|[A-Za-z]+ et al\.,? \d{4})\]', text)
    unique_citations = list(set(citations))

    # Extract Key Research Metrics (e.g. percentages, F1, accuracy, BLEU, latency)
    metric_matches = re.findall(r'(\b(?:accuracy|F1(?:-score)?|BLEU|ROUGE|perplexity|latency|speedup|AUC)\b[^.,;\n]{1,60})', text, re.IGNORECASE)
    metrics_found = [m.strip() for m in metric_matches[:6]]

    # Rigor & Scientific Confidence Score
    # Evaluates presence of abstract, methodology, results, limitations, and citations
    rigor_points = 0
    if sections.get("Abstract & Objectives"):
        rigor_points += 20
    if sections.get("Methodology & Architecture"):
        rigor_points += 25
    if sections.get("Experimental Results"):
        rigor_points += 25
    if sections.get("Limitations & Threats"):
        rigor_points += 15
    if len(unique_citations) > 3:
        rigor_points += 15

    rigor_score = min(100, max(20, rigor_points))
    if rigor_score >= 80:
        rigor_level = "High Rigor (Full Empirical Structure)"
    elif rigor_score >= 50:
        rigor_level = "Moderate Rigor"
    else:
        rigor_level = "Preliminary / Short Format"

    return {
        "domain": "academic",
        "rigor_score": rigor_score,
        "rigor_level": rigor_level,
        "citations_count": len(citations),
        "unique_citations_sample": unique_citations[:10],
        "extracted_metrics": metrics_found,
        "sections": {k: v[:3] for k, v in sections.items()}
    }


def analyze_domain(text: str, domain: str = "auto") -> Dict[str, Any]:
    """
    Run domain-specific intelligence analysis (Legal, Academic, or Auto-detect).
    """
    if domain == "auto":
        # Auto-detect domain
        legal_terms = ["agreement", "party", "parties", "indemnify", "liability", "governing law", "confidentiality", "contract"]
        academic_terms = ["abstract", "paper", "methodology", "dataset", "baseline", "results", "et al.", "experiment", "neural"]

        text_lower = text.lower()
        legal_score = sum(1 for t in legal_terms if t in text_lower)
        academic_score = sum(1 for t in academic_terms if t in text_lower)

        domain = "legal" if legal_score >= academic_score else "academic"

    if domain == "legal":
        return analyze_legal_document(text)
    else:
        return analyze_academic_document(text)
