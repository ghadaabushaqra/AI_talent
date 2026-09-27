"""Validated, stateful slot collection for conversational workforce workflows."""
from __future__ import annotations

import re
from typing import Any

from app.cv_extraction import normalize_skills

TEAM_INTENT = re.compile(
    r"\b(?:build|form|assemble|create|need|make|want)\b[^.?!]*\bteam\b|"
    r"\bteam builder\b|(?:فريق|تيم)", re.I
)
OTHER_INTENT = re.compile(
    r"\b(?:find|rank|match|search)\b[^.?!]*\b(?:candidates?|employees?)\b|"
    r"(?:مرشح|موظف|كانديديت)", re.I
)
GAP_INTENT = re.compile(r"\b(?:skill gap|skills? missing|missing skills?|upskill|gap analysis)\b|(?:فجوة|ناقصه من مهارات|ناقص من مهارات|تحليل مهارات)", re.I)
TRAINING_INTENT = re.compile(r"\b(?:training|courses?|learn|upskilling)\b|(?:تدريب|دورات|كورس|تعلم)", re.I)
MATCH_INTENT = re.compile(r"\b(?:candidates?|matching|rank matches|find employees?|search employees?)\b|(?:مرشحين|مرشح|كانديديت|مطابقه|مطابقة)", re.I)
SKILL_MARKER = re.compile(
    r"\b(?:skills?|requiring|requires|required|technologies|tech stack)\b|"
    r"(?:مهارات|المهارات|سكيلز|يتطلب|مطلوب|التقنيات)", re.I
)
TEAM_SIZE = re.compile(
    r"\b(\d+)\s*[- ]?\s*(?:person|people|members?|employees?)\b|"
    r"\bteam\s+of\s+(\d+)\b|"
    r"(?:فريق|تيم)\s*(?:من|عدده|عدد)?\s*(\d+)\b|"
    r"\b(\d+)\s*(?:أشخاص|اشخاص|موظفين|أفراد|افراد)\b", re.I
)
DURATION_MONTHS = re.compile(
    r"\b(\d+)\s*[- ]?\s*(?:months?|mos?)\b|"
    r"\b(?:for|duration(?:\s+of)?)\s+(\d+)\s*[- ]?\s*(?:months?|mos?)\b|"
    r"\b(\d+)\s*(?:أشهر|اشهر|شهور|شهر)\b", re.I
)
WORD_SIZES = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "واحد": 1, "اثنين": 2, "ثلاثة": 3, "اربعة": 4, "أربعة": 4,
    "خمسة": 5, "ستة": 6, "سبعة": 7, "ثمانية": 8, "تسعة": 9, "عشرة": 10,
}


def is_team_request(text: str) -> bool:
    return bool(TEAM_INTENT.search(text or ""))


def is_other_workflow(text: str) -> bool:
    return bool(OTHER_INTENT.search(text or "")) and not is_team_request(text)


def detect_intent(text: str) -> str | None:
    if is_team_request(text):
        return "team_builder"
    if GAP_INTENT.search(text or ""):
        return "skill_gap"
    if TRAINING_INTENT.search(text or ""):
        return "training"
    if MATCH_INTENT.search(text or ""):
        return "candidate_matching"
    return None


def parse_pool(text: str) -> str | None:
    low = (text or "").lower()
    if re.search(r"\b(?:candidates?|external applicants?)\b|(?:مرشحين|مرشح|كانديديت)", low):
        return "candidates"
    if re.search(r"\b(?:employees?|internal staff)\b|(?:موظفين|موظف)", low):
        return "employees"
    return None


def parse_result_count(text: str, *, answering_count: bool = False) -> int | None:
    low = (text or "").lower()
    found = re.search(r"\btop\s+(\d+)\b|\b(?:find|show|rank|return|list)\s+(?:me\s+)?(?:the\s+)?(?:best\s+)?(\d+)\s+(?:candidates?|employees?|people|matches)\b", low)
    if found:
        return int(next(group for group in found.groups() if group))
    if answering_count:
        bare = re.fullmatch(r"\s*(\d+)\s*", low)
        if bare:
            return int(bare.group(1))
    return None


def parse_employee_id(text: str) -> str | None:
    found = re.search(r"\bE\d{5}\b", text or "", re.I)
    return found.group(0).upper() if found else None


def mentioned_role(text: str, roles: list[dict]) -> dict | None:
    low = (text or "").lower()
    return next((role for role in roles if role["id"].lower() in low or role["name"].lower() in low), None)


def parse_team_size(text: str, *, answering_size: bool = False) -> int | None:
    found = TEAM_SIZE.search(text or "")
    if found:
        return int(next(group for group in found.groups() if group))
    low = (text or "").lower()
    for word, number in WORD_SIZES.items():
        if re.search(r"(?<!\w)" + re.escape(word) + r"(?!\w)", low) and (
            answering_size or re.search(r"\b(?:team|people|employees?|members?)\b|(?:فريق|تيم|اشخاص|أشخاص)", low)
        ):
            return number
    if answering_size:
        bare = re.fullmatch(r"\s*(\d+)\s*", text or "")
        if bare:
            return int(bare.group(1))
    return None


def parse_duration_months(text: str, *, answering_duration: bool = False) -> int | None:
    found = DURATION_MONTHS.search(text or "")
    if found:
        return int(next(group for group in found.groups() if group))
    low = (text or "").lower()
    for word, number in WORD_SIZES.items():
        if re.search(r"(?<!\w)" + re.escape(word) + r"(?!\w)", low) and re.search(r"\bmonths?\b|(?:أشهر|اشهر|شهور|شهر)", low):
            return number
    if answering_duration:
        bare = re.fullmatch(r"\s*(\d+)\s*", text or "")
        if bare:
            return int(bare.group(1))
    return None


def initial_team_state(text: str, explicit_skills: list[str] | None = None) -> dict[str, Any]:
    skills = normalize_skills(text)
    confirmed = bool(explicit_skills) or (bool(SKILL_MARKER.search(text)) and bool(skills))
    return {
        "workflow": "team_builder", "status": "collecting", "project_description": text,
        "language": "ar" if re.search(r"[\u0600-\u06ff]", text) else "en",
        "team_size": parse_team_size(text),
        "duration_months": parse_duration_months(text),
        "availability_required": True,
        "required_skills": sorted(set(explicit_skills or skills)) if confirmed else [],
        "skills_confirmed": confirmed,
        "awaiting": None,
    }


def update_team_state(state: dict[str, Any], text: str, explicit_skills: list[str] | None = None) -> dict[str, Any]:
    updated = {**state}
    size = parse_team_size(text, answering_size=state.get("awaiting") == "team_size")
    if size is not None:
        updated["team_size"] = size
    duration = parse_duration_months(text, answering_duration=state.get("awaiting") == "duration_months")
    if duration is not None:
        updated["duration_months"] = duration
    skills = sorted(set(explicit_skills or normalize_skills(text)))
    if skills and (explicit_skills or state.get("awaiting") == "required_skills" or SKILL_MARKER.search(text)):
        updated["required_skills"] = skills
        updated["skills_confirmed"] = True
    return updated


def team_question(state: dict[str, Any]) -> tuple[str | None, str | None]:
    arabic = state.get("language") == "ar"
    if state.get("team_size") is None:
        return "team_size", "كم شخص بدك في الفريق؟" if arabic else "How many people should the team have?"
    if state["team_size"] < 1 or state["team_size"] > 20:
        return "team_size", "حجم الفريق لازم يكون بين 1 و20. كم شخص بدك؟" if arabic else "Team size must be between 1 and 20. How many people do you need?"
    if state.get("duration_months") is None:
        return "duration_months", "كم مدة المشروع بالأشهر؟" if arabic else "How many months will the project run?"
    if state["duration_months"] < 1 or state["duration_months"] > 36:
        return "duration_months", "مدة المشروع لازم تكون بين شهر و36 شهر. كم شهر؟" if arabic else "Project duration must be between 1 and 36 months. How many months?"
    if not state.get("skills_confirmed") or not state.get("required_skills"):
        return "required_skills", "شو المهارات المطلوبة للمشروع؟ اكتبيها صراحةً، مثل Python وNLP وRAG؛ مش رح أفترضها من عنوان المشروع." if arabic else "Which skills are required? Please list them explicitly; I won't assume them from the project title."
    return None, None


def initial_state(workflow: str, text: str, roles: list[dict], explicit_skills: list[str] | None = None) -> dict[str, Any]:
    if workflow == "team_builder":
        return initial_team_state(text, explicit_skills)
    role = mentioned_role(text, roles)
    skills = sorted(set(explicit_skills or normalize_skills(text)))
    confirmed = bool(explicit_skills) or bool(skills) and (
        workflow == "training" or bool(SKILL_MARKER.search(text)) or
        workflow == "candidate_matching" and bool(re.search(r"\bfor\s+(?:a\s+)?(?:role\s+with\s+)?(?:Python|AWS|SQL|NLP|RAG|Docker|Airflow)\b", text, re.I))
    )
    base = {
        "workflow": workflow, "status": "collecting", "original_query": text,
        "language": "ar" if re.search(r"[\u0600-\u06ff]", text) else "en",
        "awaiting": None, "required_skills": skills if confirmed else [],
        "skills_confirmed": confirmed,
    }
    if workflow == "candidate_matching":
        base.update(pool=parse_pool(text), top_k=parse_result_count(text),
            target_role=role["id"] if role else None,
            suggested_skills=role["required_skills"] if role and not confirmed else [])
    elif workflow == "skill_gap":
        base.update(employee_id=parse_employee_id(text), target_role=role["id"] if role else None)
    return base


def update_state(state: dict[str, Any], text: str, roles: list[dict], explicit_skills: list[str] | None = None) -> dict[str, Any]:
    if state["workflow"] == "team_builder":
        return update_team_state(state, text, explicit_skills)
    updated = {**state}
    role = mentioned_role(text, roles)
    if role:
        updated["target_role"] = role["id"]
        if state["workflow"] == "candidate_matching" and not state.get("skills_confirmed"):
            updated["suggested_skills"] = role["required_skills"]
    if state["workflow"] == "candidate_matching":
        pool = parse_pool(text)
        count = parse_result_count(text, answering_count=state.get("awaiting") == "top_k")
        if pool:
            updated["pool"] = pool
        if count is not None:
            updated["top_k"] = count
    if state["workflow"] == "skill_gap":
        employee_id = parse_employee_id(text)
        if employee_id:
            updated["employee_id"] = employee_id
    skills = sorted(set(explicit_skills or normalize_skills(text)))
    if state["workflow"] == "candidate_matching" and state.get("awaiting") == "required_skills" and state.get("suggested_skills"):
        if re.search(r"\b(?:use|yes|confirm|role requirements)\b|(?:نعم|استخدم|اعتمد|متطلبات الدور)", text, re.I):
            updated["required_skills"] = state["suggested_skills"]
            updated["skills_confirmed"] = True
            return updated
    if skills and (explicit_skills or state.get("awaiting") in {"required_skills", "target_requirements", "training_skills"} or SKILL_MARKER.search(text)):
        updated["required_skills"] = skills
        updated["skills_confirmed"] = True
    return updated


def next_question(state: dict[str, Any], roles: list[dict]) -> tuple[str | None, str | None]:
    if state["workflow"] == "team_builder":
        return team_question(state)
    ar = state.get("language") == "ar"
    workflow = state["workflow"]
    if workflow == "candidate_matching":
        if not state.get("pool"):
            return "pool", "بدك أبحث بين المرشحين ولا الموظفين الحاليين؟" if ar else "Should I search candidates or current employees?"
        if not state.get("skills_confirmed") or not state.get("required_skills"):
            if state.get("suggested_skills"):
                skills = ", ".join(state["suggested_skills"])
                return "required_skills", (f"وجدت متطلبات الدور في الكتالوج: {skills}. أستخدمها، ولا بدك تحددي مهارات مختلفة؟" if ar else f"The role catalog lists: {skills}. Should I use those requirements, or would you like to specify different skills?")
            return "required_skills", "شو المهارات المطلوبة للمطابقة؟" if ar else "Which skills are required for this match?"
        if state.get("top_k") is None:
            return "top_k", "كم نتيجة بدك أرجع؟" if ar else "How many results would you like?"
        if not 1 <= state["top_k"] <= 20:
            return "top_k", "عدد النتائج لازم يكون بين 1 و20. كم نتيجة بدك؟" if ar else "Please choose a result count between 1 and 20."
    elif workflow == "skill_gap":
        if not state.get("employee_id"):
            return "employee_id", "ما رقم الموظف؟ استخدمي رقمًا موجودًا مثل E02387." if ar else "What is the employee ID? Use an existing ID such as E02387."
        if not state.get("target_role") and not state.get("required_skills"):
            role_names = ", ".join(role["name"] for role in roles)
            return "target_requirements", (f"ما الدور المستهدف أو المهارات المطلوبة؟ الأدوار المتاحة: {role_names}." if ar else f"What is the target role or required skill set? Available roles: {role_names}.")
    elif workflow == "training":
        if not state.get("required_skills"):
            return "training_skills", "لأي مهارة أو مهارات بدك تدريب؟" if ar else "Which skills do you want training for?"
    return None, None
