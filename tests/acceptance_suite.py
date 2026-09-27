"""Runnable end-to-end acceptance checks for the documented workforce flows."""
import sys
import uuid
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import Query, ask, evaluation, health, list_candidates, match, person_profile, rag, team


def run():
    checks=[]
    status=health()
    assert status['status']=='ok' and status['database']['connected']
    assert status['database']['source_of_truth']=='SQL records'
    checks.append('active SQL database health')

    directory=list_candidates(person_type='candidates',skill='AWS',limit=5)
    assert len(directory)<=5 and all('AWS' in row['skills'] for row in directory)
    checks.append('candidate directory contract and filters')

    profile=person_profile('C0600')
    assert profile['type']=='candidate' and profile['profile']['id']=='C0600' and profile['source']['evidence']
    checks.append('traceable person profile')

    ranked=match(Query(query='Find the top 3 candidates for a Senior Data Engineer requiring Python, AWS and SQL',limit=3))
    assert len(ranked['results'])==3 and ranked['requirements']['requested_top_k']==3
    assert ranked['requirements']['required_skills']==['AWS','Python','SQL']
    assert all({'component_scores','evidence','ml_success_probability'}<=set(row) for row in ranked['results'])
    assert all(0<row['ml_success_probability']<100 for row in ranked['results'])
    checks.append('exact-count explainable candidate matching')

    built=team(Query(query='Build a 4-person team for a 6-month NLP project requiring Python, NLP, RAG and AWS',skills=['Python','NLP','RAG','AWS'],team_size=4,duration_months=6))
    assert len(built['team'])==4 and built['skill_coverage'] and built['similar_projects']
    assert all(row['availability_window']['available_for_full_period'] for row in built['team'])
    checks.append('complementary team and full-period availability')

    with patch.dict('os.environ',{'OLLAMA_API_KEY':''}):
        counted=ask(Query(query='How many employees in Data have AWS?',conversation_id=str(uuid.uuid4())))
    assert counted['count']>=0 and counted['filters']=={'department':'Data','skills':['AWS']}
    checks.append('grounded employee aggregation')

    retrieval=rag(Query(query='Senior Data Engineer Python AWS SQL',rag_limit=3))
    assert retrieval['results'] and all(row['id'] for row in retrieval['results'])
    assert retrieval['retrieval_method'].startswith('hybrid_') and retrieval['embedding_dimensions']==384
    assert retrieval['dense_available'] and all('lexical_score' in row and 'dense_score' in row for row in retrieval['results'])
    checks.append('hybrid multilingual RAG source IDs and evidence')

    declined=ask(Query(query='What is the weather tomorrow?',conversation_id=str(uuid.uuid4())))
    assert declined['tool_trace']==['scope_guard'] and not declined['results']
    checks.append('out-of-scope guard')

    report=evaluation()
    assert report['label_provenance']['human_ground_truth'] is False
    assert report['classifier']['confusion_matrix'] and report['deterministic_baseline']['confusion_matrix']
    assert report['matching_evaluation']['status']=='pending_human_review'
    assert report['simulated_hr_review']['provenance']['human_ground_truth'] is False
    assert report['simulated_hr_review']['summary']['judgements']==30
    checks.append('transparent ML/baseline evaluation')

    print(f'PASS: {len(checks)} acceptance checks')
    for item in checks: print(' - '+item)


if __name__=='__main__': run()
