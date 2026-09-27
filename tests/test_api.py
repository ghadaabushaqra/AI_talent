from unittest.mock import patch
import asyncio
import io
import json
import uuid
from fastapi import UploadFile
from sqlalchemy import delete, select
from app.main import Query, ask, health, match, rag, team, upload_cv, candidates, employees, people, employee_options, directory_options, engine, rag_engine, Record, TalentProfile, ProfileSkill
from app.ollama_cloud import grounded_answer
from app.candidate_matching import Requirements, evaluate_candidate, extract_requirements, normalize_skills
from app.cv_extraction import extract_cv_profile
def test_health(): assert health()['status']=='ok'
def test_employee_application_ids_are_unique(): assert len({row['id'] for row in employees})==len(employees)
def test_people_all_contains_both_record_types():
 rows=people(person_type='all')
 assert any(row['id'].startswith('E') for row in rows)
 assert any(row['id'].startswith(('C','UPL-')) for row in rows)
def test_people_skill_filter_is_case_insensitive_and_normalizes_aliases():
 lowercase=people(person_type='employees',skill='nlp')
 uppercase=people(person_type='employees',skill='NLP')
 machine_learning=people(person_type='employees',skill='ml')
 assert lowercase and {row['id'] for row in lowercase}=={row['id'] for row in uppercase}
 assert all('NLP' in row['skills'] for row in lowercase)
 assert machine_learning and all('Machine Learning' in row['skills'] for row in machine_learning)
def test_employee_options_are_complete_unique_and_safe_for_selection():
 rows=employee_options()
 assert len(rows)==len(employees)==1000
 assert len({row['id'] for row in rows})==len(rows)
 assert all(set(row)=={'id','name','department','availability'} for row in rows)
def test_directory_options_come_from_real_records():
 options=directory_options()
 assert {'NLP','Python','AWS'} <= set(options['skills'])
 assert set(options['departments'])=={row['department'] for row in employees if row.get('department')}
 assert set(options['availability'])=={'all','employees','candidates'}
def test_match_is_grounded():
 r=match(Query(query='Find the top 3 candidates for a Senior Data Engineer role requiring Python, AWS, and SQL. Docker is preferred.'))
 assert r['results'] and r['citations']
 assert r['requirements']['required_skills']==['AWS','Python','SQL']
 assert r['requirements']['optional_skills']==['Docker']
 assert {'component_scores','evidence','matched_required_skills'} <= r['results'][0].keys()
def test_alias_and_requirement_extraction():
 assert normalize_skills('Python3 and Amazon Web Services')==['AWS','Python']
 r=extract_requirements('Top 2 Senior Data Engineer role requiring Python and SQL. Docker is preferred.')
 assert r.top_k==2 and r.seniority=='Senior' and r.required_skills==['Python','SQL'] and r.optional_skills==['Docker']
def test_no_strong_match_is_explicit():
 r=match(Query(query='Top 2 candidates requiring Airflow'))
 assert r['no_strong_match_found'] is True
 assert len(r['results'])==2 and all(x['match_status']=='partial' for x in r['results'])
def test_requested_match_count_is_preserved():
 with patch.dict('os.environ', {'OLLAMA_API_KEY': ''}):
  for n in (2,5):
   r=ask(Query(query=f'Find {n} candidates requiring Python and AWS',conversation_id=f'test-count-{n}'))
   assert r['requirements']['requested_top_k']==n
   assert len(r['results'])==n and len(r['citations'])==n
   assert r['citations']==r['record_citations']
   assert all(source_id.startswith(('ROLE-','PRJ-','TRN-')) for source_id in r['knowledge_citations'])
   assert not set(r['record_citations']) & set(r['knowledge_citations'])
   assert f'Returned {n} of {n} requested candidates' in r['answer']
def test_match_form_limit_overrides_number_in_free_text():
 r=match(Query(query='Find the top 2 candidates requiring Python and AWS',limit=5))
 assert r['requirements']['requested_top_k']==5
 assert r['requirements']['top_k']==5
 assert len(r['results'])==5
def test_team_count_is_not_stuck_at_default_four():
 with patch.dict('os.environ',{'OLLAMA_API_KEY':''}):
  r=ask(Query(query='Build a 5-person team for 6 months requiring Python and AWS',conversation_id='test-team-count'))
 assert len(r['results'])==5 and '5 of 5 requested employees' in r['answer']
def test_availability_followup_reranks_the_full_pool():
 conversation='test-rerank-availability'
 with patch.dict('os.environ', {'OLLAMA_API_KEY': ''}):
  first=ask(Query(query='Find the top 3 candidates requiring Python and AWS',conversation_id=conversation))
  second=ask(Query(query='Now exclude anyone unavailable in the next 2 weeks',conversation_id=conversation))
 assert len(first['results'])==3 and len(second['results'])==3
 assert all(x['availability']=='available' for x in second['results'])
 assert 'cannot verify the next two weeks' in second['answer']
def test_weather_declined_by_agent_contract():
 response=ask(Query(query='What is the weather tomorrow?',conversation_id=str(uuid.uuid4())))
 assert 'outside this workforce system' in response['answer']
 assert not response['results'] and response['tool_trace']==['scope_guard']
def test_llm_cannot_change_count_or_invent_records():
 class FakeResponse(io.BytesIO):
  pass
 facts={'requirements':{'required_skills':['Python']},'results':[{'matched_required_skills':['Python'],'missing_required_skills':[]}]}
 fake=FakeResponse(json.dumps({'message':{'content':'I found 99 candidates including C9999.'}}).encode())
 with patch.dict('os.environ',{'OLLAMA_API_KEY':'test-only'}), patch('app.ollama_cloud.urlopen',return_value=fake):
  answer,meta=grounded_answer('Find 2 candidates',facts,'Returned 2 of 2 requested candidates.')
 assert answer=='Returned 2 of 2 requested candidates.'
 assert meta['provider']=='deterministic_fallback'

def test_team_dialogue_collects_size_and_skills_before_running_tool():
 conversation=str(uuid.uuid4())
 with patch.dict('os.environ',{'OLLAMA_API_KEY':''}):
  first=ask(Query(query='بدي اعمل تيم مناسب لمشروع NLP',conversation_id=conversation))
  second=ask(Query(query='4',conversation_id=conversation))
  third=ask(Query(query='6 months',conversation_id=conversation))
  fourth=ask(Query(query='Python, NLP, RAG',conversation_id=conversation))
 assert first['needs_clarification'] and first['missing_fields']==['team_size'] and first['results']==[]
 assert second['needs_clarification'] and second['missing_fields']==['duration_months'] and second['results']==[]
 assert third['needs_clarification'] and third['missing_fields']==['required_skills'] and third['results']==[]
 assert not fourth['needs_clarification'] and len(fourth['results'])==4
 assert fourth['agent_state']['required_skills']==['NLP','Python','RAG'] and fourth['agent_state']['duration_months']==6
 assert fourth['similar_projects'] and 'retrieve_organizational_knowledge' in fourth['tool_trace']

def test_matching_dialogue_collects_role_requirements_and_count():
 conversation=str(uuid.uuid4())
 with patch.dict('os.environ',{'OLLAMA_API_KEY':''}):
  first=ask(Query(query='Find candidates for a Senior Data Engineer',conversation_id=conversation))
  second=ask(Query(query='Use role requirements',conversation_id=conversation))
  third=ask(Query(query='3',conversation_id=conversation))
 assert first['missing_fields']==['required_skills'] and not first['results']
 assert second['missing_fields']==['top_k'] and not second['results']
 assert len(third['results'])==3 and third['requirements']['required_skills']==['AWS','Data Engineering','Docker','Python','SQL']

def test_gap_and_training_dialogues_ask_for_missing_inputs():
 with patch.dict('os.environ',{'OLLAMA_API_KEY':''}):
  gap_id=str(uuid.uuid4())
  first_gap=ask(Query(query='Analyze skill gap for E02387',conversation_id=gap_id))
  final_gap=ask(Query(query='Senior Data Engineer',conversation_id=gap_id))
  training_id=str(uuid.uuid4())
  first_training=ask(Query(query='Find training',conversation_id=training_id))
  final_training=ask(Query(query='AWS and Docker',conversation_id=training_id))
 assert first_gap['missing_fields']==['target_requirements'] and not first_gap['results']
 assert final_gap['agent_decision']['validated_action']=='analyze_gap' and final_gap['results']
 assert first_training['missing_fields']==['training_skills'] and not first_training['results']
 assert final_training['agent_decision']['validated_action']=='search_training' and final_training['results']

def test_cv_extraction_preserves_structured_evidence_without_invention():
 text='''Education Details\n2019 Bachelor of Engineering Example University\nProjects\nProject: Claims NLP Assistant using Python and AWS.\nCertifications\nAWS Certified Developer 2022\nSkill Details\nPython- Exprience - 24 months'''
 profile=extract_cv_profile(text,candidate_id='TEST-1',name='Test Candidate',source_label='test resume')
 assert {'AWS','NLP','Python'} <= set(profile['skills'])
 assert profile['education'][0]['degree']=='Bachelor of Engineering'
 assert 'Bachelor of Engineering' in profile['education'][0]['evidence']
 assert profile['projects'][0]['title'].startswith('Claims NLP Assistant')
 assert 'Project: Claims NLP Assistant' in profile['projects'][0]['evidence']
 assert 'Project : Claims NLP Assistant' not in profile['projects'][0]['evidence']
 assert profile['raw_resume_text']==text
 assert profile['certifications'][0]['name'].startswith('AWS Certified Developer')
 assert profile['availability']=='unknown' and profile['field_status']['availability']=='unknown'
 assert profile['availability_evidence'] is None

def test_unknown_candidate_availability_is_not_scored_as_zero():
 class NeutralModel:
  def predict_proba(self, rows): return [[0.5,0.5] for _ in rows]
 candidate=extract_cv_profile('Python and AWS. 4 years of experience.',candidate_id='TEST-2',name='Unknown Availability',source_label='test resume')
 result=evaluate_candidate(candidate,Requirements(required_skills=['Python','AWS']),NeutralModel())
 assert result['availability']=='unknown'
 assert result['component_scores']['availability_score']['status']=='not_assessed'
 assert result['component_scores']['availability_score']['score'] is None
 assert result['assessed_weight'] < 1

def test_uploaded_cv_is_saved_as_matchable_candidate():
 upload=UploadFile(filename='candidate.txt',file=io.BytesIO(b'Python AWS SQL. 5 years of experience. Available immediately.'))
 result=asyncio.run(upload_cv(upload)); candidate_id=result['saved_candidate_id']
 try:
  assert result['stored'] and result['structured_profile']['availability']=='available'
  assert any(row['id']==candidate_id for row in candidates)
  from sqlalchemy.orm import Session
  with Session(engine) as session:
   assert session.scalar(select(Record).where(Record.id==f'candidate_upload:{candidate_id}'))
   assert session.scalar(select(TalentProfile).where(TalentProfile.external_id==candidate_id))
 finally:
  candidates[:]=[row for row in candidates if row['id']!=candidate_id]
  from sqlalchemy.orm import Session
  with Session(engine) as session:
   profile=session.scalar(select(TalentProfile).where(TalentProfile.external_id==candidate_id))
   if profile: session.execute(delete(ProfileSkill).where(ProfileSkill.profile_pk==profile.pk))
   session.execute(delete(TalentProfile).where(TalentProfile.external_id==candidate_id))
   session.execute(delete(Record).where(Record.id==f'candidate_upload:{candidate_id}'));session.commit()

def test_team_builder_uses_full_period_availability_and_similar_projects():
 result=team(Query(query='Six-month NLP delivery requiring Python, NLP, RAG and AWS',skills=['Python','NLP','RAG','AWS'],team_size=4,duration_months=6,availability_required=True))
 assert result['returned_team_size']==4 and result['similar_projects']
 assert result['skill_coverage'] and result['project_start'] < result['project_end']
 assert all(member['availability_window']['available_for_full_period'] for member in result['team'])
 assert set(skill for member in result['team'] for skill in member['assigned_team_skills'])=={'Python','NLP','RAG','AWS'}
 assert {'find_similar_projects','check_availability','evaluate_historical_projects'} <= set(result['tool_trace'])
 assert all(project['id'].startswith('PRJ-') and project['evidence'] for project in result['similar_projects'])

def test_rag_returns_traceable_source_ids_and_agent_uses_them():
 retrieval=rag(Query(query='Senior Data Engineer Python AWS SQL',rag_limit=5))
 assert retrieval['results'] and all(row['id'] and row['source_type'] and row['evidence'] for row in retrieval['results'])
 assert retrieval['retrieval_method'].startswith('hybrid_')
 assert all({'hybrid_score','dense_score','lexical_score'} <= row.keys() for row in retrieval['results'])
 with patch.dict('os.environ',{'OLLAMA_API_KEY':''}):
  response=ask(Query(query='Find the top 2 candidates for a Senior Data Engineer requiring Python AWS and SQL',conversation_id=str(uuid.uuid4())))
 assert response['knowledge'] and any(source['source_type']=='role' for source in response['knowledge'])
 assert 'retrieve_organizational_knowledge' in response['tool_trace']

def test_multilingual_hybrid_rag_maps_arabic_query_to_grounded_english_role():
 assert rag_engine.status()['dense_available'] is True
 retrieval=rag_engine.search('ما المهارات المطلوبة لمهندس بيانات أول؟',limit=3,source_types={'role'})
 assert retrieval['embedding_model']=='intfloat/multilingual-e5-small'
 assert retrieval['results'][0]['id']=='ROLE-001'
 assert all(row['source_type']=='role' for row in retrieval['results'])
 assert retrieval['results'][0]['dense_score'] > 0 and retrieval['results'][0]['lexical_score'] > 0
