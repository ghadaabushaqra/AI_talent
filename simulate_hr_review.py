"""Create an explicitly disclosed AI-simulated HR relevance review.

This is presentation evidence, not a substitute for independent human review.
Judgements use only structured, cited CV facts and never use ML probability or
the final weighted score as inputs.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

ROOT=Path(__file__).parent
API="http://localhost:8765"
CASES=[
    {"case_id":"SIM-HR-001","role":"Senior Data Engineer","required_skills":["Python","AWS","SQL"],"minimum_experience_years":4,"top_k":3},
    {"case_id":"SIM-HR-002","role":"NLP Engineer","required_skills":["Python","NLP","AWS"],"minimum_experience_years":3,"top_k":3},
    {"case_id":"SIM-HR-003","role":"Cloud DevOps Engineer","required_skills":["AWS","Docker","Python"],"minimum_experience_years":3,"top_k":3},
]

def post_match(case):
    skills=", ".join(case["required_skills"])
    query=f"Find the top 10 candidates for a {case['role']} requiring {skills}. Minimum {case['minimum_experience_years']} years experience."
    body=json.dumps({"query":query,"pool":"candidates","limit":10}).encode()
    request=Request(API+"/match",data=body,headers={"Content-Type":"application/json"},method="POST")
    with urlopen(request,timeout=60) as response: return query,json.load(response)

def judgement(row,case):
    missing=sorted(set(case["required_skills"])-set(row["matched_required_skills"]))
    years=row.get("experience_years")
    if missing:
        relevance=0;decision="not_relevant";reason=f"Missing required skills: {', '.join(missing)}."
    elif years is None:
        relevance=None;decision="insufficient_evidence";reason="All required skills are documented, but experience evidence is unavailable."
    elif years<case["minimum_experience_years"]:
        relevance=0;decision="not_relevant";reason=f"Documented experience ({years} years) is below the {case['minimum_experience_years']}-year minimum."
    else:
        relevance=1;decision="relevant";reason=f"All {len(case['required_skills'])} required skills and {years} documented years meet the hard requirements."
    return {"rank":row["rank"],"candidate_id":row["candidate_id"],"decision":decision,"relevance":relevance,"reason":reason,
        "documented_facts":{"matched_required_skills":row["matched_required_skills"],"missing_required_skills":missing,"experience_years":years,"project_status":row["evidence"]["projects"]["status"]},
        "source_reference":row["citation"]}

def metrics(judgements,k):
    top=[row for row in judgements[:k] if row["relevance"] is not None]
    precision=sum(row["relevance"] for row in top)/len(top) if top else None
    gains=[row["relevance"] or 0 for row in judgements]
    dcg=sum(gain/math.log2(index+2) for index,gain in enumerate(gains[:k]))
    ideal=sorted(gains,reverse=True)
    idcg=sum(gain/math.log2(index+2) for index,gain in enumerate(ideal[:k]))
    return {f"precision@{k}":None if precision is None else round(precision,3),f"ndcg@{k}":round(dcg/idcg,3) if idcg else None,"assessed_at_k":len(top),"ranking_metric_status":"calculated" if idcg else "not_applicable_no_relevant_candidate_in_reviewed_set"}

def main():
    reviewed=[]
    for case in CASES:
        query,result=post_match(case)
        judgements=[judgement(row,case) for row in result["results"]]
        reviewed.append({**case,"job_brief":query,"judgements":judgements,"metrics":metrics(judgements,case["top_k"])})
    report={
        "provenance":{"reviewer_type":"AI-simulated HR reviewer","human_ground_truth":False,"generated_at":datetime.now(timezone.utc).isoformat(),
            "required_disclosure":"These labels were produced by an AI applying a disclosed HR screening rubric; they are demonstration evidence, not independent human validation."},
        "rubric":{"relevant":"All required skills are documented and documented experience meets the minimum.","not_relevant":"At least one hard skill is missing or documented experience is below the minimum.","insufficient_evidence":"The CV does not contain enough evidence; this is not converted to a negative label.","excluded_inputs":["weighted match score","ML success probability","candidate name"]},
        "cases":reviewed,
    }
    precisions=[case["metrics"][f"precision@{case['top_k']}"] for case in reviewed if case["metrics"][f"precision@{case['top_k']}"] is not None]
    ndcgs=[case["metrics"][f"ndcg@{case['top_k']}"] for case in reviewed if case["metrics"][f"ndcg@{case['top_k']}"] is not None]
    report["summary"]={"cases":len(reviewed),"judgements":sum(len(case["judgements"]) for case in reviewed),"mean_precision_at_3":round(sum(precisions)/len(precisions),3) if precisions else None,"mean_ndcg_at_3_on_evaluable_cases":round(sum(ndcgs)/len(ndcgs),3) if ndcgs else None,"evaluable_ndcg_cases":len(ndcgs),"no_relevant_candidate_cases":len(reviewed)-len(ndcgs)}
    (ROOT/"data"/"simulated_hr_evaluation.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report["summary"],indent=2))

if __name__=="__main__": main()
