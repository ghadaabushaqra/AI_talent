"""Evidence-preserving CV extraction shared by imports and uploaded resumes.

The extractor is intentionally conservative: structured values are emitted only
when the original resume contains supporting text. Missing sections remain
``unknown`` unless the resume explicitly says that the item is absent.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


SKILL_TAXONOMY = {
    "Python": ("python", "python 3", "python3"),
    "AWS": ("aws", "amazon web services"),
    "SQL": ("sql", "pl-sql", "mysql", "sqlserver"),
    "Docker": ("docker",),
    "Airflow": ("airflow",),
    "Machine Learning": ("ml", "machine learning"),
    "NLP": ("nlp", "natural language processing"),
    "RAG": ("rag", "retrieval augmented generation"),
    "Spark": ("spark", "apache spark"),
    "Azure": ("azure",),
    "Tableau": ("tableau",),
    "Power BI": ("power bi",),
    "Java": ("java", "core java"),
    "Git": ("git", "github"),
    "FastAPI": ("fastapi",),
    "React": ("react", "react.js", "reactjs"),
    "Data Engineering": ("data engineering", "data engineer"),
}

DEGREE_PATTERN = re.compile(
    r"\b(?:B\.?E\.?|B\.?Tech|B\.?Sc|BSc|BCA|BBA|B\.M\.M|Bachelor(?:'s)?(?:\s+of\s+(?:Engineering|Technology|Science|Arts|Commerce|Business Administration))?|"
    r"M\.?E\.?|M\.?Tech|M\.?Sc|MSc|MCA|MBA|Master(?:'s)?(?:\s+of\s+(?:Engineering|Technology|Science|Arts|Commerce|Business Administration))?|"
    r"Ph\.?D|Doctorate|Post\s+Graduate\s+Diploma|Diploma|LLB|H\.?S\.?C|S\.?S\.?C)\b",
    re.I,
)
INSTITUTION_PATTERN = re.compile(
    r"\b(?:[A-Z][A-Za-z.&'-]*\s+){0,7}(?:University|College|Institute|School|Board|Polytechnic)\b(?:[^\n,;]*)",
    re.I,
)
CERT_PATTERN = re.compile(
    r"\b(?:AWS Certified[^\n,;]*|Microsoft Certified[^\n,;]*|Azure Certified[^\n,;]*|"
    r"Google Cloud Certified[^\n,;]*|PMP\b|PRINCE2\b|Scrum Master\b|"
    r"Certified [A-Za-z0-9+.#/& -]{3,90})",
    re.I,
)
EXPLICIT_ABSENCE = {
    "education": re.compile(r"\b(?:no|without)\s+(?:formal\s+)?education\b", re.I),
    "projects": re.compile(r"\bno\s+(?:professional\s+|academic\s+)?projects?\b", re.I),
    "certifications": re.compile(r"\b(?:no|without)\s+(?:professional\s+)?certifications?\b", re.I),
    "experience": re.compile(r"\b(?:no|without)\s+(?:professional\s+|work\s+)?experience\b", re.I),
}


def clean_evidence(value: str, limit: int = 520) -> str:
    """Collapse whitespace without changing the words found in the source."""
    return " ".join((value or "").replace("\x00", " ").split())[:limit]


def _source_units(text: str) -> list[str]:
    """Create stable source units even when a resume was flattened to one line."""
    normalized = (text or "").replace("\r", "\n")
    normalized = re.sub(
        r"(?i)((?:Education Details|Skill Details|Company Details|Certifications?|Projects?|Achievements/Tasks)\b\s*:?|description\s*-|company\s*-)",
        r"\n\1\n",
        normalized,
    )
    units = []
    for line in normalized.split("\n"):
        line = clean_evidence(line)
        if not line:
            continue
        # Long flattened lines are split at sentence/bullet boundaries only.
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])|\s*[•\u2022]\s*", line)
        units.extend(clean_evidence(part) for part in parts if clean_evidence(part))
    return units


def _evidence_for_match(text: str, match: re.Match[str], radius: int = 150) -> str:
    start = max(0, match.start() - radius)
    end = min(len(text), match.end() + radius)
    return clean_evidence(text[start:end])


def normalize_skills(text: str) -> list[str]:
    low = text or ""
    return sorted(
        canonical
        for canonical, aliases in SKILL_TAXONOMY.items()
        if any(re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", low, re.I) for alias in aliases)
    )


def extract_skills(text: str) -> tuple[list[str], dict[str, list[dict[str, Any]]]]:
    evidence: dict[str, list[dict[str, Any]]] = {}
    for canonical, aliases in SKILL_TAXONOMY.items():
        snippets: list[dict[str, Any]] = []
        for alias in aliases:
            for match in re.finditer(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", text or "", re.I):
                snippet = _evidence_for_match(text, match)
                if snippet and all(item["evidence"] != snippet for item in snippets):
                    snippets.append({"evidence": snippet, "source": "resume", "confidence": 1.0})
                if len(snippets) >= 3:
                    break
            if len(snippets) >= 3:
                break
        if snippets:
            evidence[canonical] = snippets
    return sorted(evidence), evidence


def _field_status(field: str, text: str, entries: list[dict[str, Any]]) -> str:
    if entries:
        return "documented"
    if EXPLICIT_ABSENCE[field].search(text or ""):
        return "confirmed_absent"
    return "unknown"


def extract_education(text: str) -> tuple[list[dict[str, Any]] | None, str]:
    entries: list[dict[str, Any]] = []
    for unit in _source_units(text):
        degree = DEGREE_PATTERN.search(unit)
        if not degree:
            continue
        year_match = re.search(r"\b(?:19|20)\d{2}\b", unit)
        institution_match = INSTITUTION_PATTERN.search(unit)
        entry = {
            "degree": clean_evidence(degree.group(0), 120),
            "field_of_study": None,
            "institution": clean_evidence(institution_match.group(0), 180) if institution_match else None,
            "year": int(year_match.group(0)) if year_match else None,
            "evidence": clean_evidence(unit),
            "source": "resume",
            "confidence": 1.0,
        }
        if all(existing["evidence"] != entry["evidence"] for existing in entries):
            entries.append(entry)
    status = _field_status("education", text, entries)
    return (entries or None), status


def extract_certifications(text: str) -> tuple[list[dict[str, Any]] | None, str]:
    entries: list[dict[str, Any]] = []
    for unit in _source_units(text):
        for match in CERT_PATTERN.finditer(unit):
            evidence = clean_evidence(unit)
            entry = {
                "name": clean_evidence(match.group(0), 180),
                "issuer": None,
                "year": int(year.group(0)) if (year := re.search(r"\b(?:19|20)\d{2}\b", unit)) else None,
                "evidence": evidence,
                "source": "resume",
                "confidence": 1.0,
            }
            if all(existing["evidence"] != evidence for existing in entries):
                entries.append(entry)
    status = _field_status("certifications", text, entries)
    return (entries or None), status


def _project_title(unit: str) -> str | None:
    explicit = re.search(r"\bprojects?\s*[:\-]\s*([^.;]{3,140})", unit, re.I)
    if explicit:
        return clean_evidence(explicit.group(1), 140)
    letters="".join(character for character in unit if character.isalpha())
    if "project" in unit.lower() and len(unit) <= 180 and letters and letters.upper()==letters:
        title = re.sub(r"(?i)\bprojects?\b\s*[:\-]?", "", unit).strip(" :-")
        return clean_evidence(title, 140) or None
    return None


def extract_projects(text: str) -> tuple[list[dict[str, Any]] | None, str]:
    units = _source_units(text)
    entries: list[dict[str, Any]] = []
    for index, unit in enumerate(units):
        if not re.search(r"\bprojects?\b", unit, re.I):
            continue
        # Keep a small contiguous source block. Every word remains traceable to the CV.
        evidence = clean_evidence(" ".join(units[index : index + 6]))
        entry = {
            "title": _project_title(evidence),
            "description": evidence,
            "skills": normalize_skills(evidence),
            "evidence": evidence,
            "source": "resume",
            "confidence": 1.0,
        }
        if evidence and all(existing["evidence"] != evidence for existing in entries):
            entries.append(entry)
        if len(entries) >= 8:
            break
    status = _field_status("projects", text, entries)
    return (entries or None), status


def extract_experience(text: str) -> tuple[float | None, list[dict[str, Any]] | None, str]:
    years: list[tuple[float, str, str]] = []
    for match in re.finditer(r"(\d+(?:\.\d+)?)\+?\s+years?\s+(?:of\s+)?experience", text or "", re.I):
        years.append((float(match.group(1)), _evidence_for_match(text, match), "explicit_total"))
    for match in re.finditer(r"(?:experience|exprience)\s*[-:]?\s*(\d+(?:\.\d+)?)\s+months?", text or "", re.I):
        years.append((round(float(match.group(1)) / 12, 1), _evidence_for_match(text, match), "documented_duration"))
    entries: list[dict[str, Any]] = []
    for unit in _source_units(text):
        company = re.search(r"(?i)\bcompany\s*-\s*(.+)", unit)
        if company:
            entries.append({
                "organization": clean_evidence(company.group(1), 180),
                "role": None,
                "dates": None,
                "evidence": clean_evidence(unit),
                "source": "resume",
                "confidence": 1.0,
            })
    best = max(years, key=lambda item: item[0]) if years else None
    if best:
        entries.insert(0, {
            "organization": None,
            "role": None,
            "dates": None,
            "duration_years": best[0],
            "calculation_method": best[2],
            "evidence": best[1],
            "source": "resume",
            "confidence": 1.0,
        })
    entries = entries[:12]
    status = "documented" if entries else ("confirmed_absent" if EXPLICIT_ABSENCE["experience"].search(text or "") else "unknown")
    return (best[0] if best else None), (entries or None), status


def extract_availability(text: str) -> tuple[str, dict[str, Any] | None, str]:
    patterns = (
        ("available", re.compile(r"\b(?:available immediately|immediately available|currently available)\b", re.I)),
        ("unavailable", re.compile(r"\b(?:not available|currently unavailable)\b", re.I)),
    )
    for value, pattern in patterns:
        match = pattern.search(text or "")
        if match:
            evidence = {"value": value, "evidence": _evidence_for_match(text, match), "source": "resume", "confidence": 1.0}
            return value, evidence, "documented"
    return "unknown", None, "unknown"


def extract_cv_profile(
    text: str,
    *,
    candidate_id: str,
    name: str,
    source_category: str = "Uploaded CV",
    source_label: str,
) -> dict[str, Any]:
    """Return a match-ready profile whose claims all retain resume evidence."""
    skills, skill_evidence = extract_skills(text)
    experience_years, experience, experience_status = extract_experience(text)
    education, education_status = extract_education(text)
    certifications, certification_status = extract_certifications(text)
    projects, project_status = extract_projects(text)
    availability, availability_evidence, availability_status = extract_availability(text)
    return {
        "id": candidate_id,
        "name": name,
        "source_category": source_category,
        "skills": skills,
        "skill_evidence": skill_evidence,
        "experience_years": experience_years,
        "experience": experience,
        "experience_evidence": [item["evidence"] for item in (experience or []) if item.get("evidence")] or None,
        "education": education,
        "certifications": certifications,
        "projects": projects,
        "availability": availability,
        "availability_evidence": availability_evidence,
        "field_status": {
            "skills": "documented" if skills else "unknown",
            "experience": experience_status,
            "education": education_status,
            "certifications": certification_status,
            "projects": project_status,
            "availability": availability_status,
        },
        "raw_resume_text": text,
        "evidence": source_label,
    }
