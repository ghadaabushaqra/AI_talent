"""Generate transparent evaluation artifacts without inventing human labels."""
import json
from pathlib import Path

from app.ml_evaluation import evaluate_success_model

ROOT=Path(__file__).parent
DATA=ROOT/'data'

def load(name):
    return json.loads((DATA/name).read_text(encoding='utf-8'))

def main():
    report=evaluate_success_model(load('candidates.json'),load('assignment_outcomes.json'))
    (DATA/'evaluation_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    review_cases=[
        {"case_id":"MATCH-001","job_brief":"Senior Data Engineer requiring Python, AWS and SQL","pool":"candidates","top_k":3,"reviewer_relevance":{},"status":"pending_human_review"},
        {"case_id":"MATCH-002","job_brief":"NLP Engineer requiring Python, NLP, RAG and FastAPI","pool":"candidates","top_k":3,"reviewer_relevance":{},"status":"pending_human_review"},
        {"case_id":"MATCH-003","job_brief":"Cloud DevOps Engineer requiring AWS, Docker, Git and Python","pool":"employees","top_k":3,"reviewer_relevance":{},"status":"pending_human_review"},
    ]
    (DATA/'matching_ground_truth_template.json').write_text(json.dumps({"instructions":"An HR/domain reviewer must label returned profile IDs as relevant (1) or not relevant (0). Do not auto-fill these judgements.","cases":review_cases},indent=2),encoding='utf-8')
    print('Wrote evaluation_report.json and matching_ground_truth_template.json')

if __name__=='__main__': main()
