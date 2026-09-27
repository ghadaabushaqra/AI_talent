"""FastAPI service: matching, ML benchmark, RAG retrieval and grounded agent traces."""
from pathlib import Path
from collections import Counter
from itertools import zip_longest
import json, re, io, os, uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, String, Text, Integer, ForeignKey, select, delete, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session
from app.candidate_matching import extract_requirements, normalize_skills, rank_candidates, train_success_model
from app.cv_extraction import extract_cv_profile
from app.ollama_cloud import grounded_answer, plan_workflow
from app.agent_dialogue import detect_intent, initial_state, update_state, next_question, parse_duration_months
from app.workforce_tools import build_complementary_team
from app.ml_evaluation import evaluate_success_model
from app.rag_service import HybridRAG

ROOT=Path(__file__).resolve().parents[1]; D=ROOT/'data'
DATABASE_URL=os.getenv('DATABASE_URL','sqlite:///'+str(ROOT/'talentflow.db'))
class Base(DeclarativeBase): pass
class Record(Base):
 __tablename__='records'; id:Mapped[str]=mapped_column(String(32),primary_key=True); kind:Mapped[str]=mapped_column(String(24)); payload:Mapped[str]=mapped_column(Text)
class TalentProfile(Base):
 __tablename__='talent_profiles'; pk:Mapped[int]=mapped_column(primary_key=True); external_id:Mapped[str]=mapped_column(String(32),index=True); profile_type:Mapped[str]=mapped_column(String(16)); name:Mapped[str]=mapped_column(String(160)); department:Mapped[str|None]=mapped_column(String(100),nullable=True); availability:Mapped[str]=mapped_column(String(24)); experience_years:Mapped[int]=mapped_column(Integer); evidence:Mapped[str]=mapped_column(Text)
class ProfileSkill(Base):
 __tablename__='profile_skills'; id:Mapped[int]=mapped_column(primary_key=True); profile_pk:Mapped[int]=mapped_column(ForeignKey('talent_profiles.pk')); skill:Mapped[str]=mapped_column(String(100),index=True)
class TrainingCourse(Base):
 __tablename__='training_courses'; id:Mapped[str]=mapped_column(String(32),primary_key=True); title:Mapped[str]=mapped_column(String(240)); skill:Mapped[str]=mapped_column(String(100),index=True); duration_hours:Mapped[int]=mapped_column(Integer); url:Mapped[str]=mapped_column(Text)
class Project(Base):
 __tablename__='projects'; id:Mapped[str]=mapped_column(String(32),primary_key=True); name:Mapped[str]=mapped_column(String(180)); duration_months:Mapped[int]=mapped_column(Integer); team_size:Mapped[int]=mapped_column(Integer); description:Mapped[str]=mapped_column(Text)
class AuditEvent(Base):
 __tablename__='audit_events'; id:Mapped[int]=mapped_column(primary_key=True); action:Mapped[str]=mapped_column(String(80)); request_payload:Mapped[str]=mapped_column(Text); citation_ids:Mapped[str]=mapped_column(Text)
class ChatTurn(Base):
 __tablename__='chat_turns'; id:Mapped[int]=mapped_column(primary_key=True); conversation_id:Mapped[str]=mapped_column(String(80),index=True); user_query:Mapped[str]=mapped_column(Text); response_payload:Mapped[str]=mapped_column(Text)
engine=create_engine(DATABASE_URL, pool_pre_ping=True)
def load(n): return json.loads((D/n).read_text(encoding='utf8'))
employees,candidates,projects,courses,outcomes=map(load,['employees.json','candidates.json','projects.json','training_catalog.json','assignment_outcomes.json'])
target_roles=load('target_roles.json')
def seed_database():
 Base.metadata.create_all(engine)
 with Session(engine) as s:
  # Source-derived records are reproducible. User-uploaded CV records persist.
  uploaded=[json.loads(row.payload) for row in s.scalars(select(Record).where(Record.kind=='candidate_upload'))]
  s.execute(delete(Record).where(Record.kind!='candidate_upload'))
  for kind,rows in [('employee',employees),('candidate',candidates),('project',projects),('course',courses),('outcome',outcomes)]: s.add_all([Record(id=f'{kind}:{i}:{x["id"] if "id" in x else x["candidate_id"]}',kind=kind,payload=json.dumps(x)) for i,x in enumerate(rows)])
  # Refresh the relational read model so regenerated evidence is never stale.
  s.execute(delete(ProfileSkill));s.execute(delete(TalentProfile));s.execute(delete(TrainingCourse));s.execute(delete(Project))
  profiles=[]
  for typ,rows in [('employee',employees),('candidate',candidates+uploaded)]:
   for p in rows:
    profile=TalentProfile(external_id=p['id'],profile_type=typ,name=p.get('name') or f'{typ.title()} {p["id"]}',department=p.get('department'),availability=p.get('availability','unknown'),experience_years=int(round(p.get('experience_years') or 0)),evidence=p['evidence']); s.add(profile); profiles.append((profile,p))
  s.flush()
  s.add_all([ProfileSkill(profile_pk=profile.pk,skill=skill) for profile,p in profiles for skill in p['skills']])
  s.add_all([TrainingCourse(id=c['id'],title=c['title'],skill=c['skill'],duration_hours=c['duration_hours'],url=c['url']) for c in courses])
  s.add_all([Project(id=p['id'],name=p['name'],duration_months=p['duration_months'],team_size=p['team_size'],description=p['description']) for p in projects]); s.commit()
  result={k:[json.loads(x.payload) for x in s.scalars(select(Record).where(Record.kind==k))] for k in ['employee','candidate','project','course','outcome']}
  result['candidate'].extend(uploaded)
  return result
db=seed_database(); employees,candidates,projects,courses,outcomes=db['employee'],db['candidate'],db['project'],db['course'],db['outcome']
def records_by_kind(kind:str) -> list[dict]:
 """Read operational records from the configured SQL database."""
 with Session(engine) as s:
  rows=[json.loads(row.payload) for row in s.scalars(select(Record).where(Record.kind==kind))]
  if kind=='candidate':
   rows.extend(json.loads(row.payload) for row in s.scalars(select(Record).where(Record.kind=='candidate_upload')))
  return rows

def person_by_id(person_id:str) -> tuple[dict|None,str|None]:
 for kind in ('employee','candidate'):
  person=next((row for row in records_by_kind(kind) if row.get('id')==person_id),None)
  if person: return person,kind
 return None,None
AL={'ml':'Machine Learning','machine learning':'Machine Learning','natural language processing':'NLP','nlp':'NLP','python':'Python','sql':'SQL','aws':'AWS','docker':'Docker','react':'React','fastapi':'FastAPI','rag':'RAG','spark':'Spark','azure':'Azure','tableau':'Tableau','power bi':'Power BI','java':'Java','git':'Git','data engineering':'Data Engineering'}
def extract(text): return normalize_skills(text)
candidate_match_model=train_success_model(candidates,outcomes)
def one(p, req):
 hit=sorted(set(p['skills'])&set(req)); gaps=sorted(set(req)-set(p['skills'])); weighted=round(55*len(hit)/max(1,len(req))+30*min(1,p.get('experience_years',0)/8)+15*(p['availability']=='available'),1)
 probability=round(float(candidate_match_model.predict_proba([[len(p.get('skills') or []),p.get('experience_years') or 0,len(p.get('past_projects') or []),0,0]])[0][1])*100,1)
 return {'id':p['id'],'name':p['name'],'job_title':p.get('job_title',p.get('source_category','Candidate')),'weighted_score':weighted,'ml_success_probability':probability,'skills_matched':hit,'skill_gaps':gaps,'experience_years':p.get('experience_years',0),'availability':p['availability'],'past_projects':p.get('past_projects',[]),'citation':p['evidence'],'explanation':f"{len(hit)} of {len(req)} required skills documented; {p.get('experience_years',0)} years experience; availability is {p['availability']}."}
def ranking(pool,req,n=5): return sorted([one(p,req) for p in pool],key=lambda x:x['weighted_score'],reverse=True)[:n]
rag_documents=(
 [{'id':role['id'],'source_type':'role','title':role['name'],'text':f"{role['name']}. Required skills: {', '.join(role['required_skills'])}. Minimum experience: {role['minimum_experience']} years."} for role in target_roles]
 +[{'id':p['id'],'source_type':'project','title':p['name'],'text':f"{p['name']}. {p['description']} Required skills: {', '.join(p.get('required_skills',[]))}. Duration: {p['duration_months']} months."} for p in projects]
 +[{'id':c['id'],'source_type':'training','title':c['title'],'text':f"{c['title']}. Skill: {c['skill']}. Duration: {c['duration_hours']} hours. URL: {c['url']}"} for c in courses]
)
rag_engine=HybridRAG(rag_documents)
app=FastAPI(title='TalentFlow AI',version='1.0.0');app.add_middleware(CORSMiddleware,allow_origins=['*'],allow_methods=['*'],allow_headers=['*'])
class Query(BaseModel): query:str=''; skills:list[str]=[]; limit:int=5; team_size:int=4; duration_months:int=0; start_date:str=''; availability_required:bool=True; employee_id:str=''; target_role:str=''; pool:str='candidates'; conversation_id:str='default'; rag_limit:int=3
def audit(action,q,citations):
 payload=q if isinstance(q,dict) else q.model_dump()
 with Session(engine) as s: s.add(AuditEvent(action=action,request_payload=json.dumps(payload),citation_ids=json.dumps(citations)));s.commit()
def remember(q,response):
 with Session(engine) as s: s.add(ChatTurn(conversation_id=q.conversation_id,user_query=q.query,response_payload=json.dumps(response)));s.commit()
@app.get('/health')
def health():
 try:
  with Session(engine) as s:
   s.execute(text('SELECT 1'))
   counts={kind:len(records_by_kind(kind)) for kind in ('employee','candidate','project','course')}
  return {'status':'ok','database':{'dialect':engine.dialect.name,'connected':True,'source_of_truth':'SQL records'},'rag':rag_engine.status(),'records':{'employees':counts['employee'],'candidates':counts['candidate'],'projects':counts['project'],'courses':counts['course']}}
 except Exception as exc:
  raise HTTPException(503,detail={'status':'unhealthy','database_connected':False,'error':type(exc).__name__})
@app.get('/dashboard')
def dashboard_summary():
 employee_rows=records_by_kind('employee');candidate_rows=records_by_kind('candidate')
 departments=Counter(row.get('department') or 'Unassigned' for row in employee_rows)
 skills=Counter(skill for row in employee_rows for skill in row.get('skills',[]))
 availability=Counter(row.get('availability','unknown') for row in employee_rows)
 with Session(engine) as s:
  recent=[{'action':event.action,'citations':len(json.loads(event.citation_ids))} for event in s.scalars(select(AuditEvent).order_by(AuditEvent.id.desc()).limit(5))]
 return {
  'counts':{'employees':len(employee_rows),'candidates':len(candidate_rows),'courses':len(records_by_kind('course')),'projects':len(records_by_kind('project'))},
  'departments':[{'name':name,'count':count,'percent':round(100*count/max(1,len(employee_rows)),1)} for name,count in departments.most_common(7)],
  'top_skills':[{'name':name,'count':count,'percent':round(100*count/max(1,len(employee_rows)),1)} for name,count in skills.most_common(8)],
  'availability':dict(availability),'recent_activity':recent,
  'database':{'dialect':engine.dialect.name,'connected':True},
 }
@app.get('/candidates')
def list_candidates(person_type:str='candidates',skill:str|None=None,department:str|None=None,availability:str|None=None,limit:int=100):
 """Documented directory contract; defaults to candidates and can explicitly include employees."""
 if person_type not in {'candidates','employees','all'}: raise HTTPException(400,"person_type must be candidates, employees, or all")
 pool=(records_by_kind('candidate') if person_type=='candidates' else records_by_kind('employee') if person_type=='employees' else records_by_kind('candidate')+records_by_kind('employee'))
 return filter_people(pool,skill,department,availability)[:max(1,min(500,limit))]
def filter_people(pool:list[dict],skill:str|None=None,department:str|None=None,availability:str|None=None,search:str|None=None):
 requested_skills=normalize_skills(skill or '')
 if skill and not requested_skills: requested_skills=[skill.strip()]
 requested_skills={value.casefold() for value in requested_skills}
 requested_department=(department or '').strip().casefold()
 requested_availability=(availability or '').strip().casefold()
 search_term=(search or '').strip().casefold()
 result=[]
 for person in pool:
  person_skills={value.casefold() for value in person.get('skills',[])}
  haystack=' '.join([person.get('name',''),person.get('id',''),person.get('department',''),' '.join(person.get('skills',[]))]).casefold()
  if requested_skills and not requested_skills.issubset(person_skills): continue
  if requested_department and (person.get('department') or '').casefold()!=requested_department: continue
  if requested_availability and (person.get('availability') or '').casefold()!=requested_availability: continue
  if search_term and search_term not in haystack: continue
  result.append(person)
 return result
@app.get('/people')
def people(person_type:str='all',skill:str|None=None,department:str|None=None,availability:str|None=None,search:str|None=None):
 if person_type not in {'candidates','employees','all'}: raise HTTPException(400,"person_type must be candidates, employees, or all")
 if person_type=='employees': pool=records_by_kind('employee')
 elif person_type=='candidates': pool=records_by_kind('candidate')
 else:
  employee_rows,candidate_rows=records_by_kind('employee'),records_by_kind('candidate')
  pool=[row for pair in zip_longest(employee_rows,candidate_rows) for row in pair if row is not None]
 return filter_people(pool,skill,department,availability,search)[:200]
@app.get('/employee-options')
def employee_options():
 rows=records_by_kind('employee')
 return sorted(({'id':row['id'],'name':row['name'],'department':row.get('department') or 'Unassigned','availability':row.get('availability','unknown')} for row in rows),key=lambda row:(row['name'].lower(),row['id']))
@app.get('/directory-options')
def directory_options():
 employee_rows=records_by_kind('employee');candidate_rows=records_by_kind('candidate')
 return {
  'skills':sorted({skill for row in employee_rows+candidate_rows for skill in row.get('skills',[])},key=str.casefold),
  'departments':sorted({row['department'] for row in employee_rows if row.get('department')},key=str.casefold),
  'availability':{
   'all':sorted({row.get('availability','unknown') for row in employee_rows+candidate_rows}),
   'employees':sorted({row.get('availability','unknown') for row in employee_rows}),
   'candidates':sorted({row.get('availability','unknown') for row in candidate_rows}),
  },
 }
@app.get('/people/{person_id}')
def person_profile(person_id:str):
 person,kind=person_by_id(person_id)
 if not person: raise HTTPException(404,'Person ID not found')
 safe={key:value for key,value in person.items() if key!='raw_resume_text'}
 return {'type':kind,'profile':safe,'source':{'id':person_id,'evidence':person.get('evidence'),'field_status':person.get('field_status',{})}}
@app.get('/target-roles')
def roles(): return target_roles
@app.post('/upload-cv')
async def upload_cv(file:UploadFile=File(...)):
 blob=await file.read()
 filename=file.filename or 'uploaded-cv.txt'; lower=filename.lower()
 if lower.endswith('.pdf'):
  from pypdf import PdfReader
  raw='\n'.join(page.extract_text() or '' for page in PdfReader(io.BytesIO(blob)).pages)
  supported_format='PDF'
 elif lower.endswith('.docx'):
  from docx import Document
  document=Document(io.BytesIO(blob));raw='\n'.join(p.text for p in document.paragraphs)
  supported_format='DOCX'
 elif lower.endswith('.txt'):
  raw=blob.decode('utf8',errors='ignore');supported_format='text'
 else:
  raise HTTPException(400,'Supported CV formats are PDF, DOCX, and TXT')
 if not raw.strip(): raise HTTPException(400,'No readable resume text was found')
 candidate_id='UPL-'+uuid.uuid4().hex[:10].upper()
 profile=extract_cv_profile(raw,candidate_id=candidate_id,name=f'Uploaded Candidate {candidate_id[-4:]}',source_category='Uploaded CV',source_label=f'Uploaded CV {filename}')
 with Session(engine) as s:
  s.add(Record(id=f'candidate_upload:{candidate_id}',kind='candidate_upload',payload=json.dumps(profile)))
  talent=TalentProfile(external_id=candidate_id,profile_type='candidate',name=profile['name'],department=None,availability=profile['availability'],experience_years=int(round(profile.get('experience_years') or 0)),evidence=profile['evidence']);s.add(talent);s.flush()
  s.add_all([ProfileSkill(profile_pk=talent.pk,skill=skill) for skill in profile['skills']]);s.commit()
 candidates.append(profile)
 audit('upload-cv',{'filename':filename,'candidate_id':candidate_id,'field_status':profile['field_status']},[candidate_id])
 public_profile={key:value for key,value in profile.items() if key!='raw_resume_text'}
 public_profile['raw_text_length']=len(raw);public_profile['supported_format']=supported_format
 return {'filename':filename,'saved_candidate_id':candidate_id,'stored':True,'structured_profile':public_profile,'tool_trace':['extract_cv_information','normalize_skills','preserve_source_evidence','save_candidate_profile']}
@app.post('/match')
def match(q:Query):
 requirements=extract_requirements(q.query,q.limit)
 if 'limit' in q.model_fields_set:
  requirements.requested_top_k=q.limit
  requirements.top_k=max(1,min(20,q.limit))
 if q.skills: requirements.required_skills=sorted(set(q.skills))
 if q.target_role:
  role=next((r for r in target_roles if r['id']==q.target_role or r['name'].lower()==q.target_role.lower()),None)
  if role and not requirements.required_skills: requirements.required_skills=role['required_skills']; requirements.role=role['name']
 pool=records_by_kind('employee') if q.pool=='employees' else records_by_kind('candidate')
 if requirements.availability_constraint=='available': pool=[p for p in pool if p.get('availability')=='available']
 results,total=rank_candidates(pool,requirements,candidate_match_model)
 cited=[r['candidate_id'] for r in results]
 tool_execution=[{'tool':'extract_job_requirements','input':q.query,'result':requirements.payload()}, {'tool':'search_candidates','input':{'pool':q.pool},'result':{'evaluated':total}}, {'tool':'calculate_skill_match','input':requirements.payload(),'result':{'shortlisted':len(results)}}, {'tool':'rank_candidates','input':{'top_k':requirements.top_k},'result':cited}]
 audit('match',{'request':q.model_dump(),'tool_execution':tool_execution},cited)
 strong_count=sum(1 for row in results if row['strong_match'])
 return {'requirements':requirements.payload(),'required_skills':requirements.required_skills or [],'total_candidates_evaluated':total,'results':results,'strong_match_count':strong_count,'no_strong_match_found':strong_count==0,'message':'No strong match found from documented candidate evidence; showing the best partial matches with their gaps.' if strong_count==0 else None,'tool_trace':[x['tool'] for x in tool_execution],'tool_execution':tool_execution,'citations':cited}
@app.post('/team-builder')
def team(q:Query):
 req=q.skills or extract(q.query);duration=q.duration_months or parse_duration_months(q.query) or 6
 employee_rows,project_rows,course_rows=records_by_kind('employee'),records_by_kind('project'),records_by_kind('course')
 result=build_complementary_team(employees=employee_rows,projects=project_rows,courses=course_rows,project_brief=q.query,required_skills=req,team_size=q.team_size,duration_months=duration,start_date=q.start_date or None,availability_required=q.availability_required)
 knowledge=retrieve_knowledge(q.query+' '+ ' '.join(req),5,{'project'})
 cited=list(dict.fromkeys([member['id'] for member in result['team']]+[project['id'] for project in result['similar_projects']]+[course['id'] for course in result['training_recommendations']]+knowledge['citations']))
 tools=['extract_job_requirements','retrieve_organizational_knowledge','find_similar_projects','search_employees','check_availability','evaluate_historical_projects','build_team','identify_skill_gaps','search_training']
 tool_execution=[
  {'tool':'extract_job_requirements','input':q.query,'result':{'required_skills':req,'team_size':q.team_size,'duration_months':duration,'start_date':result['project_start']}},
  {'tool':'retrieve_organizational_knowledge','input':q.query,'result':knowledge['citations']},
  {'tool':'find_similar_projects','input':{'skills':req},'result':[project['id'] for project in result['similar_projects']]},
  {'tool':'check_availability','input':{'start':result['project_start'],'end':result['project_end']},'result':{'eligible':len(employee_rows)-len(result['excluded_for_availability']),'excluded':len(result['excluded_for_availability'])}},
  {'tool':'build_team','input':{'team_size':q.team_size},'result':[member['id'] for member in result['team']]},
  {'tool':'identify_skill_gaps','input':req,'result':result['skill_gaps']},
  {'tool':'search_training','input':result['skill_gaps'],'result':[course['id'] for course in result['training_recommendations']]},
 ]
 audit('team-builder',{'request':q.model_dump(),'tool_execution':tool_execution,'result':{'team_ids':[member['id'] for member in result['team']],'coverage_percent':result['coverage_percent'],'skill_gaps':result['skill_gaps'],'similar_project_ids':[project['id'] for project in result['similar_projects']]}},cited)
 return {**result,'rag_evidence':knowledge['results'],'tool_trace':tools,'tool_execution':tool_execution,'citations':cited}
@app.post('/skill-gap')
def gap(q:Query):
 p=next((x for x in records_by_kind('employee') if x['id']==q.employee_id),None)
 if not p: raise HTTPException(404,'Employee ID not found')
 role=next((x for x in target_roles if x['id']==q.target_role or x['name'].lower()==q.target_role.lower()),None);required=q.skills or (role['required_skills'] if role else extract(q.target_role)); r=one(p,required);audit('skill-gap',q,[p['id']]);return {'employee':r,'required_skills':required,'training_recommendations':[c for c in records_by_kind('course') if c['skill'] in r['skill_gaps']],'tool_trace':['search_employees','identify_skill_gaps','search_training'],'citations':[p['id']]}
def retrieve_knowledge(query:str,limit:int=3,source_types:set[str]|None=None):
 return rag_engine.search(query,limit,source_types)
@app.post('/rag/search')
def rag(q:Query):
 result=retrieve_knowledge(q.query,q.rag_limit);audit('rag-search',{'query':q.query,'source_ids':result['citations']},result['citations']);return result
@app.post('/ask')
def ask(q:Query):
 query=q.query.strip()
 with Session(engine) as session:
  previous=session.scalar(select(ChatTurn).where(ChatTurn.conversation_id==q.conversation_id).order_by(ChatTurn.id.desc()))
 old=json.loads(previous.response_payload) if previous else {}
 prior_state=old.get('agent_state')

 if re.search(r'\b(?:cancel|start over|reset)\b|(?:إلغاء|الغاء|ابدأ من جديد)',query,re.I):
  response={'answer':'Okay, I cleared the current request. What workforce task would you like to start?' if not re.search(r'[\u0600-\u06ff]',query) else 'تمام، ألغيت الطلب الحالي. شو المهمة الجديدة؟','results':[],'citations':[],'tool_trace':['conversation_memory','reset_workflow'],'agent_state':None}
  audit('ask_reset',q,[]);remember(q,response);return response

 if re.search(r'\b(?:weather|forecast|temperature)\b|(?:الطقس|حالة الجو|درجة الحرارة)',query,re.I):
  response={'answer':'Weather is outside this workforce system. I can help with candidates, teams, skill gaps, or training.' if not re.search(r'[\u0600-\u06ff]',query) else 'الطقس خارج نطاق نظام القوى العاملة. بقدر أساعدك بالمرشحين، الفرق، فجوات المهارات، أو التدريب.','results':[],'citations':[],'tool_trace':['scope_guard'],'agent_state':prior_state}
  audit('ask_scope_guard',q,[]);remember(q,response);return response

 # Deterministic directory aggregation is a lookup, not a ranking workflow.
 if re.search(r'\b(?:how many|count)\b|(?:كم عدد|عدد الموظفين)',query,re.I) and re.search(r'\b(?:employees?|staff)\b|(?:موظف|موظفين)',query,re.I):
  employee_rows=records_by_kind('employee'); query_skills=extract(query)
  departments=sorted({row.get('department') for row in employee_rows if row.get('department')},key=len,reverse=True)
  department=next((value for value in departments if re.search(r'(?<!\w)'+re.escape(value)+r'(?!\w)',query,re.I)),None)
  requested_department=re.search(r'\bin\s+([A-Za-z &]+?)\s+(?:have|with|who)\b',query,re.I)
  if not department and requested_department: department=requested_department.group(1).strip().title()
  matches=[row for row in employee_rows if (not department or row.get('department')==department) and all(skill in row.get('skills',[]) for skill in query_skills)]
  citations=[row['id'] for row in matches[:20]]
  criteria=', '.join(([department] if department else [])+query_skills) or 'the requested criteria'
  response={'answer':f"Found {len(matches)} employees matching {criteria}.",'count':len(matches),'results':matches[:20],'citations':citations,'tool_trace':['parse_directory_filters','search_employees','count_records'],'agent_state':None,'needs_clarification':False,'filters':{'department':department,'skills':query_skills}}
  audit('ask_people_count',{'query':query,'filters':response['filters'],'count':len(matches)},citations);remember(q,response);return response

 # A completed matching task can be refined by re-running the full eligible pool.
 if 'exclude' in query.lower() and any(x in query.lower() for x in ['unavailable','not available','availability']) and old.get('requirements'):
  prior=old['requirements']; pool=(prior_state or {}).get('pool') or ('employees' if (old.get('results') or [{}])[0].get('id','').startswith('E') else 'candidates')
  reranked=match(Query(query=f"Find the top {prior.get('top_k',3)} {pool} requiring {', '.join(prior.get('required_skills',[]))}; available",skills=prior.get('required_skills',[]),limit=prior.get('top_k',3),pool=pool,conversation_id=q.conversation_id))
  results=reranked['results']; cited=reranked['citations']
  response={'answer':f"Returned {len(results)} of {prior.get('requested_top_k') or prior.get('top_k',3)} requested {pool} after checking stored current availability. The dataset has no future scheduling, so I cannot verify the next two weeks.",'requirements':reranked['requirements'],'results':results,'citations':cited,'tool_trace':['conversation_memory','check_availability','rank_candidates'],'no_strong_match_found':reranked['no_strong_match_found'],'agent_state':prior_state}
  audit('ask_followup',q,cited);remember(q,response);return response

 plan,planner=plan_workflow(query,prior_state)
 explicit_intent=detect_intent(query)
 if prior_state and prior_state.get('status')=='collecting' and not explicit_intent:
  workflow=prior_state['workflow']
 elif explicit_intent:
  workflow=explicit_intent
 elif prior_state and prior_state.get('status')=='complete' and re.search(r'\b(?:also|change|make it|add|instead)\b|(?:كمان|غير|غيّر|اضف|أضف)',query,re.I):
  workflow=prior_state['workflow']
 else:
  workflow=plan.get('intent') if plan.get('intent') not in (None,'unknown') else None
 if workflow not in {'team_builder','candidate_matching','skill_gap','training'}:
  response={'answer':'I can help with candidate matching, team building, skill gaps, or training. Which one do you need?' if not re.search(r'[\u0600-\u06ff]',query) else 'بقدر أساعدك بمطابقة المرشحين، بناء فريق، فجوة المهارات، أو التدريب. أي مهمة بدك؟','results':[],'citations':[],'tool_trace':['llm_plan' if plan else 'intent_fallback','ask_clarification'],'planner':planner,'needs_clarification':True,'agent_state':prior_state}
  audit('ask_clarification',{'query':query,'planner':planner},[]);remember(q,response);return response

 if prior_state and prior_state.get('workflow')==workflow and (prior_state.get('status')=='collecting' or not explicit_intent):
  state=update_state(prior_state,query,target_roles,q.skills)
 else:
  state=initial_state(workflow,query,target_roles,q.skills)
 if 'team_size' in q.model_fields_set and workflow=='team_builder': state['team_size']=q.team_size
 if 'duration_months' in q.model_fields_set and workflow=='team_builder' and q.duration_months: state['duration_months']=q.duration_months
 if workflow=='team_builder': state['availability_required']=q.availability_required
 if 'limit' in q.model_fields_set and workflow=='candidate_matching': state['top_k']=q.limit
 if 'pool' in q.model_fields_set and workflow=='candidate_matching': state['pool']=q.pool
 if q.employee_id and workflow=='skill_gap': state['employee_id']=q.employee_id
 if q.target_role and workflow in {'skill_gap','candidate_matching'}: state['target_role']=q.target_role
 if state.get('target_role') and not any(role['id']==state['target_role'] for role in target_roles): state['target_role']=None
 if workflow=='skill_gap' and state.get('employee_id') and not any(p['id']==state['employee_id'] for p in employees):
  state['employee_id']=None
  invalid_id=True
 else: invalid_id=False

 missing,question=next_question(state,target_roles)
 if missing:
  state['status']='collecting';state['awaiting']=missing
  if invalid_id: question=('I could not find that employee ID. ' if state['language']=='en' else 'ما لقيت رقم الموظف هذا. ')+question
  response={'answer':question,'results':[],'citations':[],'tool_trace':['llm_plan' if plan else 'intent_fallback','conversation_memory','validate_required_fields','ask_clarification'],'planner':planner,'agent_decision':{'proposed_action':plan.get('action'),'validated_action':'ask_'+missing},'needs_clarification':True,'missing_fields':[missing],'agent_state':state}
  audit('ask_clarification',{'query':query,'workflow':workflow,'missing':missing,'planner':planner},[]);remember(q,response);return response

 state['status']='complete';state['awaiting']=None
 if workflow=='team_builder':
  result=team(Query(query=state['project_description'],skills=state['required_skills'],team_size=state['team_size'],duration_months=state['duration_months'],start_date=q.start_date,availability_required=state.get('availability_required',True),conversation_id=q.conversation_id))
  record_citations=[row['id'] for row in result['team']];knowledge_citations=[item['id'] for item in result['rag_evidence']]
  response={'answer':f"Built a team of {len(result['team'])} of {state['team_size']} requested employees with {result['coverage_percent']}% documented skill coverage for {result['project_start']} to {result['project_end']}." if result['strong_team'] else result['message'],'results':result['team'],'citations':record_citations,'record_citations':record_citations,'knowledge_citations':knowledge_citations,'knowledge':result['rag_evidence'],'tool_trace':['llm_plan' if plan else 'intent_fallback','validate_required_fields']+result['tool_trace'],'skill_gaps':result['skill_gaps'],'training_recommendations':result['training_recommendations'],'similar_projects':result['similar_projects'],'skill_coverage':result['skill_coverage'],'team_explanation':result['explanation'],'availability_window':{'start':result['project_start'],'end':result['project_end']},'planner':planner,'agent_decision':{'proposed_action':plan.get('action'),'validated_action':'build_team'},'agent_state':state}
 elif workflow=='candidate_matching':
  pool=state['pool']; count=state['top_k']; skills=state['required_skills']
  canonical=f"Find the top {count} {pool} requiring {', '.join(skills)}. {state['original_query']}"
  result=match(Query(query=canonical,skills=skills,limit=count,pool=pool,target_role=state.get('target_role') or '',conversation_id=q.conversation_id))
  rows=result['results']; strong=result['strong_match_count']
  fallback=f"Returned {len(rows)} of {count} requested {pool}; {strong} strong and {len(rows)-strong} partial matches. Review documented evidence and gaps below."
  knowledge=retrieve_knowledge(state['original_query'],3,{'role'})
  facts={'requirements':result['requirements'],'results':[{'id':r['id'],'matched_required_skills':r['matched_required_skills'],'missing_required_skills':r['missing_required_skills']} for r in rows],'knowledge':knowledge['results']}
  answer,llm=grounded_answer(state['original_query'],facts,fallback)
  response={'answer':answer,'requirements':result['requirements'],'results':rows,'citations':result['citations'],'record_citations':result['citations'],'knowledge_citations':knowledge['citations'],'knowledge':knowledge['results'],'tool_trace':['llm_plan' if plan else 'intent_fallback','validate_required_fields']+result['tool_trace']+knowledge['tool_trace'],'no_strong_match_found':result['no_strong_match_found'],'llm':llm,'planner':planner,'agent_decision':{'proposed_action':plan.get('action'),'validated_action':'candidate_match'},'agent_state':state}
 elif workflow=='skill_gap':
  result=gap(Query(employee_id=state['employee_id'],target_role=state.get('target_role') or '',skills=[] if state.get('target_role') else state['required_skills'],conversation_id=q.conversation_id))
  knowledge=retrieve_knowledge(' '.join(result['required_skills']),5,{'role','training'})
  response={'answer':f"Compared {state['employee_id']} with the target requirements. Found {len(result['employee']['skill_gaps'])} documented skill gaps and {len(result['training_recommendations'])} catalog training options.",'results':[result['employee']],'citations':result['citations'],'record_citations':result['citations'],'knowledge_citations':knowledge['citations'],'knowledge':knowledge['results'],'tool_trace':['llm_plan' if plan else 'intent_fallback','validate_required_fields']+result['tool_trace']+knowledge['tool_trace'],'skill_gaps':result['employee']['skill_gaps'],'training_recommendations':result['training_recommendations'],'planner':planner,'agent_decision':{'proposed_action':plan.get('action'),'validated_action':'analyze_gap'},'agent_state':state}
 else:
  rows=[course for course in courses if course['skill'] in state['required_skills']]
  knowledge=retrieve_knowledge(' '.join(state['required_skills']),5,{'training'});record_citations=[course['id'] for course in rows]
  response={'answer':f"Found {len(rows)} training catalog entries for {', '.join(state['required_skills'])}. These are demo catalog entries; verify course links before use.",'results':rows,'citations':record_citations,'record_citations':record_citations,'knowledge_citations':knowledge['citations'],'knowledge':knowledge['results'],'tool_trace':['llm_plan' if plan else 'intent_fallback','validate_required_fields','search_training']+knowledge['tool_trace'],'planner':planner,'agent_decision':{'proposed_action':plan.get('action'),'validated_action':'search_training'},'agent_state':state}
 response['needs_clarification']=False
 audit_citations=list(dict.fromkeys(response.get('record_citations',response['citations'])+response.get('knowledge_citations',[])))
 audit('ask',{'query':query,'workflow':workflow,'decision':response['agent_decision'],'planner':planner,'tool_trace':response['tool_trace'],'knowledge_ids':[item['id'] for item in response.get('knowledge',[])]},audit_citations)
 remember(q,response)
 return response
@app.get('/audit-events')
def audit_events():
 with Session(engine) as s: return [{'action':e.action,'request_and_tool_results':json.loads(e.request_payload),'citations':json.loads(e.citation_ids)} for e in s.scalars(select(AuditEvent).order_by(AuditEvent.id.desc()).limit(50))]
@app.get('/evaluation')
def evaluation():
 report=evaluate_success_model(candidates,outcomes)
 simulated=D/'simulated_hr_evaluation.json'
 if simulated.exists(): report['simulated_hr_review']=json.loads(simulated.read_text(encoding='utf8'))
 return report
