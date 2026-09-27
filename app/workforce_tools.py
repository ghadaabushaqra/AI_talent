"""Deterministic project similarity, availability, and complementary team tools."""
from __future__ import annotations

from datetime import date, datetime, timedelta
import re
from typing import Any

from app.cv_extraction import normalize_skills


def _parse_date(value: str | None, fallback: date | None = None) -> date:
    if value:
        try:
            return datetime.strptime(value, "%Y-%m-%d").date()
        except ValueError:
            pass
    return fallback or date.today()


def project_window(start_date: str | None, duration_months: int) -> tuple[date, date]:
    start = _parse_date(start_date)
    duration = max(1, min(36, int(duration_months or 1)))
    return start, start + timedelta(days=duration * 30)


def check_availability(employee: dict[str, Any], start_date: str | None, duration_months: int) -> dict[str, Any]:
    """Check the full requested window against current and scheduled allocations."""
    start, end = project_window(start_date, duration_months)
    conflicts: list[dict[str, Any]] = []
    if employee.get("exit_date"):
        conflicts.append({
            "type": "employment_ended",
            "end_date": employee["exit_date"],
            "evidence": employee.get("evidence"),
        })
    for allocation in employee.get("allocations", []):
        allocation_start = _parse_date(allocation.get("start_date"), start)
        allocation_end = _parse_date(allocation.get("end_date"), allocation_start)
        if allocation_start <= end and allocation_end >= start:
            conflicts.append({
                "type": "allocation_overlap",
                "project_id": allocation.get("project_id"),
                "start_date": allocation_start.isoformat(),
                "end_date": allocation_end.isoformat(),
                "status": allocation.get("status"),
                "evidence": allocation.get("evidence"),
            })
    return {
        "available_for_full_period": not conflicts,
        "project_start": start.isoformat(),
        "project_end": end.isoformat(),
        "conflicts": conflicts,
        "evidence": employee.get("availability_evidence") or employee.get("evidence"),
    }


def _tokens(value: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[A-Za-z][A-Za-z0-9+#.-]{2,}", value or "")}


def find_similar_projects(
    project_brief: str,
    required_skills: list[str],
    projects: list[dict[str, Any]],
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Rank catalog projects using documented skill and lexical overlap."""
    required = set(required_skills)
    query_tokens = _tokens(project_brief)
    results: list[dict[str, Any]] = []
    for project in projects:
        project_skills = set(project.get("required_skills") or normalize_skills(project.get("description", "")))
        skill_similarity = len(required & project_skills) / len(required) if required else 0.0
        project_tokens = _tokens(project.get("name", "") + " " + project.get("description", ""))
        lexical_similarity = len(query_tokens & project_tokens) / len(query_tokens) if query_tokens else 0.0
        similarity = 0.85 * skill_similarity + 0.15 * lexical_similarity
        if similarity > 0:
            results.append({
                "id": project["id"],
                "name": project["name"],
                "similarity_score": round(similarity, 3),
                "similarity_percent": round(similarity * 100, 1),
                "matched_skills": sorted(required & project_skills),
                "required_skills": project.get("required_skills", []),
                "evidence": project.get("description"),
            })
    results.sort(key=lambda item: (item["similarity_score"], len(item["matched_skills"])), reverse=True)
    return results[: max(1, limit)]


def historical_project_relevance(
    employee: dict[str, Any],
    required_skills: list[str],
    projects_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    required = set(required_skills)
    evidence: list[dict[str, Any]] = []
    best = 0.0
    for project_id in employee.get("past_projects", []):
        project = projects_by_id.get(project_id)
        if not project:
            continue
        project_skills = set(project.get("required_skills", []))
        score = len(required & project_skills) / len(required) if required else 0.0
        best = max(best, score)
        evidence.append({
            "project_id": project_id,
            "project_name": project.get("name"),
            "matched_skills": sorted(required & project_skills),
            "similarity_score": round(score, 3),
            "evidence": project.get("description"),
        })
    evidence.sort(key=lambda item: item["similarity_score"], reverse=True)
    return {"score": round(best, 3), "projects": evidence}


def build_complementary_team(
    *,
    employees: list[dict[str, Any]],
    projects: list[dict[str, Any]],
    courses: list[dict[str, Any]],
    project_brief: str,
    required_skills: list[str],
    team_size: int,
    duration_months: int,
    start_date: str | None,
    availability_required: bool,
) -> dict[str, Any]:
    required = sorted(set(required_skills))
    requested_size = max(1, min(20, int(team_size or 1)))
    duration = max(1, min(36, int(duration_months or 1)))
    projects_by_id = {project["id"]: project for project in projects}
    similar = find_similar_projects(project_brief, required, projects)
    pool: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for employee in employees:
        availability = check_availability(employee, start_date, duration)
        if availability_required and not availability["available_for_full_period"]:
            excluded.append({"employee_id": employee["id"], "conflicts": availability["conflicts"]})
            continue
        relevance = historical_project_relevance(employee, required, projects_by_id)
        pool.append({"employee": employee, "availability": availability, "project_relevance": relevance})

    uncovered = set(required)
    selected: list[dict[str, Any]] = []
    while pool and len(selected) < requested_size:
        def selection_score(item: dict[str, Any]) -> tuple[float, float, float, float]:
            employee = item["employee"]
            skills = set(employee.get("skills", []))
            coverage_gain = len(skills & uncovered) / max(1, len(required))
            direct_coverage = len(skills & set(required)) / max(1, len(required))
            project_relevance = item["project_relevance"]["score"]
            experience = min(1.0, (employee.get("experience_years") or 0) / 8)
            availability = 1.0 if item["availability"]["available_for_full_period"] else 0.0
            total = 0.45 * coverage_gain + 0.20 * direct_coverage + 0.20 * project_relevance + 0.10 * experience + 0.05 * availability
            return total, coverage_gain, project_relevance, experience

        chosen = max(pool, key=selection_score)
        pool.remove(chosen)
        employee = chosen["employee"]
        employee_skills = set(employee.get("skills", []))
        matched = sorted(employee_skills & set(required))
        unique_contribution = sorted(employee_skills & uncovered)
        total, _, project_score, _ = selection_score(chosen)
        uncovered -= employee_skills
        selected.append({
            "id": employee["id"],
            "candidate_id": employee["id"],
            "name": employee["name"],
            "job_title": employee.get("job_title"),
            "department": employee.get("department"),
            "weighted_score": round(total * 100, 1),
            "score_label": "Team contribution",
            "ml_success_probability": None,
            "skills_matched": matched,
            "matched_required_skills": matched,
            "missing_required_skills": sorted(set(required) - employee_skills),
            "skill_gaps": sorted(set(required) - employee_skills),
            "unique_skill_contribution": unique_contribution,
            "experience_years": employee.get("experience_years"),
            "availability": "available for full period" if chosen["availability"]["available_for_full_period"] else "has schedule conflict",
            "availability_window": chosen["availability"],
            "project_relevance": chosen["project_relevance"],
            "past_projects": employee.get("past_projects", []),
            "citation": employee.get("evidence"),
            "explanation": (
                f"Contributes {', '.join(unique_contribution) if unique_contribution else 'supporting capacity'}; "
                f"historical project relevance {round(project_score * 100, 1)}%; "
                f"availability checked for {chosen['availability']['project_start']} to {chosen['availability']['project_end']}."
            ),
        })

    # Assign explicit skill ownership across the selected team so a complete
    # team is explainable even when one versatile member covers several skills.
    assignments = {member["id"]: [] for member in selected}
    for skill in sorted(required, key=lambda value: sum(value in member["matched_required_skills"] for member in selected)):
        capable = [member for member in selected if skill in member["matched_required_skills"]]
        if capable:
            owner = min(capable, key=lambda member: (len(assignments[member["id"]]), -member["weighted_score"]))
            assignments[owner["id"]].append(skill)
    for member in selected:
        member["assigned_team_skills"] = assignments[member["id"]]
        member["team_role"] = f"{', '.join(assignments[member['id']])} owner" if assignments[member["id"]] else "Delivery support"

    coverage = {
        skill: {
            "covered": any(skill in member["matched_required_skills"] for member in selected),
            "employee_ids": [member["id"] for member in selected if skill in member["matched_required_skills"]],
        }
        for skill in required
    }
    missing = [skill for skill, value in coverage.items() if not value["covered"]]
    training = [course for course in courses if course["skill"] in missing]
    covered_count = len(required) - len(missing)
    coverage_percent = round(100 * covered_count / max(1, len(required)))
    complete = len(selected) == requested_size and not missing
    return {
        "team": selected,
        "requested_team_size": requested_size,
        "returned_team_size": len(selected),
        "duration_months": duration,
        "project_start": project_window(start_date, duration)[0].isoformat(),
        "project_end": project_window(start_date, duration)[1].isoformat(),
        "availability_required": availability_required,
        "coverage_percent": coverage_percent,
        "skill_coverage": coverage,
        "skill_gaps": missing,
        "training_recommendations": training,
        "similar_projects": similar,
        "excluded_for_availability": excluded,
        "strong_team": complete,
        "message": None if complete else "No fully suitable team was found for the requested size, skills, and availability window; returning the strongest documented partial team.",
        "explanation": (
            f"Selected {len(selected)} employees as a complementary set covering {covered_count}/{len(required)} required skills. "
            "Selection prioritized new team skill coverage, documented similar-project experience, experience, and full-period availability."
        ),
    }
