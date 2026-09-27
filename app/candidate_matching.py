"""Deterministic, evidence-grounded candidate matching pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any

from app.cv_extraction import SKILL_TAXONOMY, normalize_skills
from app.ml_evaluation import success_features, train_success_model

TAXONOMY = SKILL_TAXONOMY
SEMANTIC_CONFIDENCE_THRESHOLD = 0.88

@dataclass
class Requirements:
    role: str | None = None
    seniority: str | None = None
    required_skills: list[str] | None = None
    optional_skills: list[str] | None = None
    minimum_experience_years: int | None = None
    education_requirements: list[str] | None = None
    certification_requirements: list[str] | None = None
    availability_constraint: str | None = None
    top_k: int = 3
    requested_top_k: int | None = None

    def payload(self) -> dict[str, Any]:
        value = asdict(self)
        value["required_skills"] = value["required_skills"] or []
        value["optional_skills"] = value["optional_skills"] or []
        value["education_requirements"] = value["education_requirements"] or []
        value["certification_requirements"] = value["certification_requirements"] or []
        return value

def extract_requirements(text: str, fallback_top_k: int = 3) -> Requirements:
    low = (text or "").lower()
    top = re.search(r"\btop\s+(\d+)\b", low) or re.search(
        r"\b(?:find|show|rank|recommend|return|list)\s+(?:me\s+)?(?:the\s+)?(?:best\s+)?(\d+)\s+(?:candidates?|employees?|people|matches)\b", low
    )
    years = re.search(r"(?:at least|minimum|min\.?|with)\s+(\d+)\+?\s+years?", low)
    seniority = next((x.title() for x in ("senior", "junior", "mid", "lead", "principal") if re.search(rf"\b{x}\b", low)), None)
    role_match = re.search(r"(?:for (?:an? )?|role(?: of)?\s+)([A-Za-z ]+?)(?:\s+(?:role|requiring|with|where|,|\.))", text or "", re.I)
    role = role_match.group(1).strip() if role_match else None
    optional_clause = re.search(r"(?:preferred|optional|nice to have)\s*[:\-]?\s*([^\.]+)", text or "", re.I)
    optional = normalize_skills(optional_clause.group(1)) if optional_clause else []
    optional += [canonical for canonical, aliases in TAXONOMY.items() if any(re.search(r"(?<!\w)"+re.escape(alias)+r"(?!\w)\s+(?:is\s+)?(?:preferred|optional|nice to have)", low) for alias in aliases)]
    optional = sorted(set(optional))
    required_clause = re.search(r"(?:requiring|requires|required\s+skills?\s*:?|must\s+have)\s+(.+?)(?:\.|$)", text or "", re.I)
    # An explicit requirement clause overrides incidental skill-like words in
    # the role title (for example Data Engineer must not silently add Data
    # Engineering when the user explicitly listed Python, AWS and SQL).
    all_skills = normalize_skills(required_clause.group(1) if required_clause else text)
    required = [skill for skill in all_skills if skill not in optional]
    availability = "available" if (re.search(r"(?<!not\s)\bavailable\b|\bunassigned\b", low) or "next 2 weeks" in low) else None
    education = re.findall(r"(?:bachelor'?s|master'?s|phd|bsc|msc)\s*(?:in\s+[A-Za-z ]+)?", text or "", re.I)
    certifications = re.findall(r"(?:aws certified|azure certified|pmp|scrum master)", text or "", re.I)
    requested_top_k = int(top.group(1)) if top else fallback_top_k
    return Requirements(role=role, seniority=seniority, required_skills=required, optional_skills=optional,
        minimum_experience_years=int(years.group(1)) if years else None,
        education_requirements=education, certification_requirements=certifications,
        availability_constraint=availability, top_k=max(1, min(20, requested_top_k)), requested_top_k=requested_top_k)

def structured_profile(candidate: dict[str, Any]) -> dict[str, Any]:
    """Preserve missing Resume Dataset fields as unknown rather than fabricated."""
    return {"candidate_id": candidate["id"], "skills": candidate.get("skills", []),
        "skill_evidence":candidate.get("skill_evidence",{}),
        "experience_years": candidate.get("experience_years"), "experience_timeline":candidate.get("experience") or candidate.get("experience_evidence"),
        "education": candidate.get("education"), "certifications": candidate.get("certifications"), "projects": candidate.get("projects") or candidate.get("past_projects") or None,
        "availability_evidence":candidate.get("availability_evidence"),
        "field_status":candidate.get("field_status",{}),
        "source_evidence": candidate.get("evidence"), "source_category": candidate.get("source_category")}

def evaluate_candidate(candidate: dict[str, Any], req: Requirements, model: Any, success_probability: float | None = None) -> dict[str, Any]:
    profile = structured_profile(candidate)
    skills = set(profile["skills"])
    required, optional = set(req.required_skills or []), set(req.optional_skills or [])
    matched_required, missing_required = sorted(skills & required), sorted(required - skills)
    matched_preferred, missing_preferred = sorted(skills & optional), sorted(optional - skills)
    required_coverage = len(matched_required) / len(required) if required else 0.0
    preferred_coverage = len(matched_preferred) / len(optional) if optional else 0.0
    # Required coverage is the full skill component when no preferred skills were requested.
    skill_score = required_coverage if not optional else 0.9 * required_coverage + 0.1 * preferred_coverage
    years = profile["experience_years"]
    experience_score = (min(1.0, years / req.minimum_experience_years) if req.minimum_experience_years else min(1.0, years / 8)) if years is not None else None
    statuses=profile.get("field_status") or {}
    projects = profile["projects"]
    project_text=" ".join(str(item.get("evidence", "")) if isinstance(item, dict) else str(item) for item in (projects or []))
    if projects is not None:
        project_score=(len(set(normalize_skills(project_text)) & required)/len(required) if required else 0.0)
    elif statuses.get("projects")=="confirmed_absent":
        project_score=0.0
    else:
        project_score=None
    availability = candidate.get("availability", "unknown")
    availability_score = 1.0 if availability == "available" else 0.0 if availability in {"allocated","unavailable"} else None
    education, certifications=profile["education"],profile["certifications"]
    requested_qualifications=(req.education_requirements or [])+(req.certification_requirements or [])
    qualification_text=" ".join(str(item.get("evidence", "")) for item in (education or [])+(certifications or []))
    qualifications_assessable=(education is not None or certifications is not None or statuses.get("education")=="confirmed_absent" or statuses.get("certifications")=="confirmed_absent")
    education_certification_score=(sum(1 for item in requested_qualifications if item.lower() in qualification_text.lower())/len(requested_qualifications)) if requested_qualifications and qualifications_assessable else None
    components={"skill_score":(skill_score,0.40),"experience_score":(experience_score,0.25),"project_score":(project_score,0.15),"availability_score":(availability_score,0.10),"education_certification_score":(education_certification_score,0.10)}
    assessed={name:(value,weight) for name,(value,weight) in components.items() if value is not None}
    weighted=sum(value*weight for value,weight in assessed.values())/sum(weight for _,weight in assessed.values())
    probability = success_probability if success_probability is not None else model.predict_proba([success_features(candidate)])[0][1]
    skill_evidence=profile.get("skill_evidence") or {}
    evidence = {"skills": [{"skill":skill,"evidence":skill_evidence.get(skill) or [{"evidence":candidate.get("evidence"),"source":"profile_record","confidence":1.0}]} for skill in matched_required + matched_preferred],
        "experience": {"value_years": years, "evidence":profile.get("experience_timeline"), "status":statuses.get("experience","documented" if years is not None else "unknown"),"source": "structured_profile"},
        "projects": {"value": projects, "status":statuses.get("projects","unknown" if projects is None else "documented")},
        "education_certifications": {"education":education,"certifications":certifications,"status":"not_assessed" if not requested_qualifications else ("documented" if qualifications_assessable else "unknown")},
        "availability": {"value": availability, "evidence":profile.get("availability_evidence"),"status":statuses.get("availability","unknown" if availability=="unknown" else "documented")}}
    strong = not missing_required and weighted >= 0.50 and (not req.minimum_experience_years or (years or 0) >= req.minimum_experience_years)
    explanation = (f"Satisfies {len(matched_required)}/{len(required)} required skills; "
        f"{len(matched_preferred)}/{len(optional)} preferred skills; {years if years is not None else 'unknown'} years experience; "
        f"availability is {availability}. Unknown components were not assessed and their weights were redistributed across documented components.")
    return {"candidate_id": candidate["id"], "id": candidate["id"], "name": candidate.get("name"), "job_title": candidate.get("source_category", "Candidate"),
        "rank": 0, "weighted_score": round(weighted * 100, 1), "ml_success_probability": round(float(probability) * 100, 1),
        "matched_required_skills": matched_required, "missing_required_skills": missing_required,
        "matched_preferred_skills": matched_preferred, "missing_preferred_skills": missing_preferred,
        "skills_matched": matched_required, "skill_gaps": missing_required,
        "component_scores": {name:{"score":None if value is None else round(value,3),"status":"not_assessed" if value is None else "assessed","weight":weight} for name,(value,weight) in components.items()}, "assessed_weight":round(sum(weight for _,weight in assessed.values()),2),
        "experience_years": years, "availability": availability, "past_projects": projects, "evidence": evidence,
        "citation": candidate.get("evidence"), "explanation": explanation, "strong_match": strong}

def rank_candidates(candidates: list[dict[str, Any]], req: Requirements, model: Any) -> tuple[list[dict[str, Any]], int]:
    # One vectorized prediction avoids invoking the calibrated model hundreds
    # of times for a single ranking request.
    probabilities = [float(row[1]) for row in model.predict_proba([success_features(candidate) for candidate in candidates])]
    evaluated = [evaluate_candidate(candidate, req, model, probability) for candidate, probability in zip(candidates, probabilities)]
    evaluated.sort(key=lambda row: (row["strong_match"], row["weighted_score"], row["ml_success_probability"]), reverse=True)
    for index, row in enumerate(evaluated, 1): row["rank"] = index
    # Always return the requested shortlist. A partial match is visibly labelled and never treated as equivalent to a strong match.
    for row in evaluated:
        row["match_status"] = "strong" if row["strong_match"] else "partial"
    return evaluated[:req.top_k], len(evaluated)
