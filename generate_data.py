"""Create privacy-safe application records from the two supplied datasets."""
import csv, json, random, re
from datetime import date, timedelta
from pathlib import Path

from app.cv_extraction import extract_cv_profile, normalize_skills

ROOT = Path(__file__).parent
random.seed(27)
SKILLS = ["Python", "SQL", "AWS", "Machine Learning", "NLP", "RAG", "Docker", "FastAPI", "React", "Data Engineering", "Spark", "Power BI", "Java", "Azure", "Tableau", "Git"]
OFFICIAL_LEARNING_URLS = {
    "Python": "https://docs.python.org/3/tutorial/",
    "SQL": "https://www.postgresql.org/docs/current/tutorial.html",
    "AWS": "https://skillbuilder.aws/",
    "Machine Learning": "https://scikit-learn.org/stable/getting_started.html",
    "NLP": "https://huggingface.co/learn/nlp-course/chapter1/1",
    "RAG": "https://python.langchain.com/docs/tutorials/rag/",
    "Docker": "https://docs.docker.com/get-started/",
    "FastAPI": "https://fastapi.tiangolo.com/tutorial/",
    "React": "https://react.dev/learn",
    "Data Engineering": "https://learn.microsoft.com/en-us/training/career-paths/data-engineer",
    "Spark": "https://spark.apache.org/docs/latest/quick-start.html",
    "Power BI": "https://learn.microsoft.com/en-us/training/powerplatform/power-bi/",
    "Java": "https://dev.java/learn/",
    "Azure": "https://learn.microsoft.com/en-us/training/azure/",
    "Tableau": "https://www.tableau.com/learn/training",
    "Git": "https://git-scm.com/docs/gittutorial",
}
ALIASES = {"ml": "Machine Learning", "machine learning": "Machine Learning", "natural language processing": "NLP", "nlp": "NLP", "python": "Python", "sql": "SQL", "aws": "AWS", "docker": "Docker", "react": "React", "fastapi": "FastAPI", "rag": "RAG", "spark": "Spark", "azure": "Azure", "tableau": "Tableau", "power bi": "Power BI", "java": "Java", "git": "Git"}

def normalize(text):
    return normalize_skills(text)

def write(name, records):
    (ROOT / "data" / name).write_text(json.dumps(records, indent=2), encoding="utf-8")

def main():
    # Candidate profiles remain anchored to actual resume text/category; names are intentionally synthetic.
    candidates = []
    with open(ROOT / "UpdatedResumeDataSet.csv", encoding="utf-8", errors="replace") as f:
        for idx, row in enumerate(csv.DictReader(f)):
            resume=row["Resume"]
            candidates.append(extract_cv_profile(
                resume,
                candidate_id=f"C{idx+1:04d}",
                name=f"Candidate {idx+1:03d}",
                source_category=row["Category"],
                source_label="Resume dataset record " + str(idx+1),
            ))
    # Deterministic, explained extensions; never overwrite the supplied HR fields.
    import openpyxl
    ws = openpyxl.load_workbook(ROOT / "Employee Sample Data - A.xlsx", data_only=True).active
    columns = [c.value for c in next(ws.iter_rows(values_only=False))]
    employees=[]
    used_employee_ids=set()
    for idx, values in enumerate(ws.iter_rows(min_row=2, values_only=True)):
        r=dict(zip(columns, values)); title=str(r.get("Job Title") or "")
        base=normalize(title + " " + str(r.get("Department") or ""))
        pool=base + random.sample(SKILLS, k=3 if len(base)<2 else 2)
        unique=sorted(set(pool))
        hire=r.get("Hire Date"); tenure=max(1, (date.today()-hire.date()).days//365) if hasattr(hire,"date") else 2
        exit_date=r.get("Exit Date")
        # Keep original HR attributes but normalize missing values and add only documented synthetic extensions.
        source_employee_id=str(r["EEID"])
        employee_id=source_employee_id
        if employee_id in used_employee_ids:
            employee_id=f"E{100000+idx:06d}"
            while employee_id in used_employee_ids:
                employee_id=f"E{int(employee_id[1:])+1:06d}"
        used_employee_ids.add(employee_id)
        past_projects=[f"PRJ-{101+((idx+j*7)%20)}" for j in range(1+(idx%3))]
        if exit_date:
            availability="unavailable";allocations=[];available_from=None
        elif idx%4==0:
            availability="allocated"
            allocation_end=date.today()+timedelta(days=30+(idx%6)*30)
            allocations=[{"project_id":past_projects[0],"start_date":(date.today()-timedelta(days=30+(idx%45))).isoformat(),"end_date":allocation_end.isoformat(),"status":"active","evidence":f"Generated allocation extension for employee {employee_id}"}]
            available_from=(allocation_end+timedelta(days=1)).isoformat()
        else:
            availability="available";available_from=date.today().isoformat();allocations=[]
            # Some currently available employees have documented future commitments.
            if idx%7==0:
                future_start=date.today()+timedelta(days=45+(idx%4)*15)
                allocations.append({"project_id":past_projects[0],"start_date":future_start.isoformat(),"end_date":(future_start+timedelta(days=90)).isoformat(),"status":"scheduled","evidence":f"Generated allocation extension for employee {employee_id}"})
        employees.append({"id":employee_id,"source_employee_id":source_employee_id,"name":r.get("Full Name") or f"Employee {employee_id}","job_title":title or "Unspecified role","department":r.get("Department") or "Unassigned","business_unit":r.get("Business Unit") or "Unassigned","hire_date":hire.date().isoformat() if hasattr(hire,"date") else None,"annual_salary":float(r.get("Annual Salary") or 0),"bonus_percent":float(r.get("Bonus %") or 0),"country":r.get("Country") or "Unknown","city":r.get("City") or "Unknown","exit_date":exit_date.date().isoformat() if hasattr(exit_date,"date") else None,"skills":unique,"experience_years":tenure,"availability":availability,"available_from":available_from,"allocations":allocations,"availability_evidence":f"Generated availability extension for employee {employee_id}","past_projects":past_projects,"evidence":f"Employee dataset record {idx+1}; source EEID {source_employee_id}; application ID {employee_id}"})
    names=["Knowledge Navigator","Data Platform Modernization","Customer Insight Studio","Cloud Migration","Talent Analytics","Fraud Detection","Document Intelligence","Service Portal","MLOps Foundation","Sales Forecasting","Quality Dashboard","Customer Support AI","Security Automation","Mobile Platform","API Gateway","Data Governance","Learning Hub","Operations Console","Insight Lakehouse","Search Modernization"]
    projects=[{"id":f"PRJ-{101+i}","name":n,"duration_months":3+i%6,"team_size":3+i%3,"required_skills":random.sample(SKILLS,4),"description":f"{n} delivery brief grounded in the synthetic project catalog."} for i,n in enumerate(names)]
    courses=[]
    for i in range(25):
      skill=SKILLS[i%len(SKILLS)]; courses.append({"id":f"TRN-{i+1:02d}","title":f"Applied {skill} {'Foundations' if i<16 else 'Practitioner'}","skill":skill,"duration_hours":12+(i%4)*4,"url":OFFICIAL_LEARNING_URLS[skill],"provider":"Official learning resource","catalog_status":"curated_link"})
    outcomes=[]
    for index,c in enumerate(candidates):
        # Availability is intentionally excluded because the source CVs do not
        # reliably document it. Deterministic noise prevents a circular label
        # that merely restates one input feature.
        skill_signal=min(1.0,len(c["skills"])/6)
        experience_signal=min(1.0,(c["experience_years"] or 0)/8)
        project_signal=1.0 if c.get("projects") else 0.0
        # A reproducible but meaningful uncertainty term prevents the proxy
        # label from being a near-perfect restatement of its input features.
        # Real assignment success also depends on factors absent from a CV.
        deterministic_noise=(((index*37)%101)/100)*0.36-0.18
        score=.60*skill_signal+.25*experience_signal+.15*project_signal+deterministic_noise
        outcomes.append({"candidate_id":c["id"],"assignment_success":int(score>=.52),"signal":"documented skills, experience, project evidence, and deterministic noise"})
    target_roles=[
      {"id":"ROLE-001","name":"Senior Data Engineer","required_skills":["Python","AWS","SQL","Data Engineering","Docker"],"minimum_experience":4},
      {"id":"ROLE-002","name":"NLP Engineer","required_skills":["Python","NLP","RAG","FastAPI"],"minimum_experience":3},
      {"id":"ROLE-003","name":"Cloud DevOps Engineer","required_skills":["AWS","Docker","Git","Python"],"minimum_experience":3},
      {"id":"ROLE-004","name":"Data Analyst","required_skills":["SQL","Python","Power BI","Tableau"],"minimum_experience":2},
    ]
    write("candidates.json",candidates); write("employees.json",employees); write("projects.json",projects); write("training_catalog.json",courses); write("assignment_outcomes.json",outcomes); write("target_roles.json",target_roles)
    print(f"Generated {len(candidates)} candidates and {len(employees)} employees.")
if __name__ == "__main__": main()
