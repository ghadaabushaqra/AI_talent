# TalentAI — AI Talent & Workforce Matching Agent

TalentAI is a complete HR decision-support prototype built with React, FastAPI, PostgreSQL, deterministic matching, machine learning, hybrid multilingual RAG, and an optional Ollama Cloud planner.

The system uses the two supplied datasets for candidate discovery, workforce search, explainable matching, complementary team building, skill-gap analysis, training recommendations, CV processing, and a grounded conversational assistant.

> TalentAI supports HR decisions; it does not replace human review. Missing CV evidence is represented as unknown and is never invented.

## Demo video

[Watch the working project demo on Google Drive](https://drive.google.com/file/d/1GK46ov3YtgyY8Xc3cXbg5qg46XqeKpQ9/view?usp=sharing)

## Main capabilities

- **Workforce Dashboard:** live dataset counts and working workflow shortcuts.
- **People Directory:** employee and candidate search with skill, department, and availability filters.
- **Person Profile:** searchable employee list with documented experience, availability, skills, projects, and source evidence.
- **Candidate Matching:** explicit result count and explainable 40/25/15/10/10 weighted scoring.
- **ML Success Probability:** displayed separately from the deterministic match score.
- **Team Builder:** complementary skill coverage, schedule availability, and past-project evidence.
- **Skill Gap & Upskilling:** compares an employee with a target role and recommends catalog training.
- **CV Processing:** PDF, DOCX, and TXT upload with structured evidence for extracted claims.
- **TalentAI Assistant:** context, clarification questions, validated tools, citations, and audit traces.
- **Hybrid multilingual RAG:** multilingual E5 embeddings, TF-IDF lexical search, and FAISS.

## Architecture

    React / Vite frontend
            │  /api
            ▼
    FastAPI application ── Custom agent ── Ollama Cloud (optional)
            │                    │
            │                    ├── deterministic workforce tools
            │                    └── hybrid E5 + TF-IDF + FAISS RAG
            ▼
    PostgreSQL reference database
            │
            └── records generated from the supplied datasets

### Technology stack

| Layer | Implementation |
|---|---|
| Frontend | React, Vite, modular responsive CSS |
| API | FastAPI, Pydantic |
| Database | PostgreSQL in Docker; SQLite is a local-development fallback |
| Matching | Deterministic weighted baseline with evidence-aware normalization |
| ML | Scikit-learn logistic regression, evaluated separately from matching |
| RAG | intfloat/multilingual-e5-small + TF-IDF + FAISS |
| Agent | Validated custom agent with optional Ollama Cloud planning |
| Deployment | Docker Compose |

## Quick start with Docker

### Prerequisites

- Docker Desktop with the WSL 2 engine running.
- At least 6 GB of available memory is recommended.

### 1. Configure the optional cloud model

The system works without an API key. To enable Ollama Cloud:

    Copy-Item .env.example .env

Place the private key only in the local .env file:

    OLLAMA_API_KEY=your_private_key
    OLLAMA_CLOUD_MODEL=gpt-oss:20b-cloud

Never commit .env.

### 2. Start the complete stack

    docker compose up --build

The first build downloads the free multilingual E5 model and can take several minutes.

### 3. Open the application

- React application: <http://localhost:5173>
- FastAPI documentation: <http://localhost:8765/docs>
- Health and active backend details: <http://localhost:8765/health>

Stop the stack with:

    docker compose down

## Local development without Docker

PostgreSQL via Docker is the reference deployment. This lightweight mode uses a generated local SQLite database:

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    python generate_data.py
    python -m uvicorn app.main:app --reload --port 8765

In another terminal:

    cd frontend
    corepack enable
    pnpm install --frozen-lockfile
    pnpm dev

## Source datasets and generated records

| File | Purpose |
|---|---|
| UpdatedResumeDataSet.csv | Original resume text used to create candidate profiles |
| Employee Sample Data - A.xlsx | Original employee data |
| generate_data.py | Deterministic cleaning and enrichment pipeline |
| data/candidates.json | Structured candidates with CV evidence |
| data/employees.json | Employee profiles and extension provenance |
| data/projects.json | Project catalog used by team matching and RAG |
| data/training_catalog.json | Training catalog used by skill-gap workflows |
| data/target_roles.json | Target role requirements |

Run python generate_data.py whenever the source datasets or extraction logic changes.

## Matching and evidence policy

The deterministic score preserves the intended component weights:

| Component | Weight |
|---|---:|
| Skills | 40% |
| Experience | 25% |
| Relevant projects | 15% |
| Availability | 10% |
| Education and certifications | 10% |

- Unknown information is marked **Not Assessed**, not scored as zero.
- Weights are redistributed only across assessable components.
- Confirmed absence and unavailable evidence remain distinct.
- ML Success Probability never changes the deterministic weighted score.
- Every returned person includes a stable record ID and source evidence.

## RAG and agent behavior

The knowledge index contains role, project, and training records. Retrieval combines:

- 75% multilingual E5 semantic similarity;
- 25% TF-IDF lexical similarity;
- normalized FAISS inner-product search over 384-dimensional vectors.

The assistant applies a scope guard, asks for missing information, executes validated tools, and separates employee/candidate record citations from RAG knowledge citations. The optional LLM proposes a workflow and helps word grounded responses; backend validation controls required fields, tools, result counts, and evidence.

## API overview

| Method | Endpoint | Purpose |
|---|---|---|
| GET | /health | Service, database, dataset, RAG, and LLM status |
| GET | /dashboard | Dashboard metrics and actionable activity |
| GET | /people | Search employees and candidates |
| GET | /people/{person_id} | Structured profile with source evidence |
| POST | /upload-cv | Parse and save a CV |
| POST | /match | Explainable candidate or employee matching |
| POST | /team-builder | Complementary team construction |
| POST | /skill-gap | Individual skill-gap analysis |
| POST | /rag/search | Raw grounded knowledge retrieval |
| POST | /ask | Conversational workforce agent |
| GET | /evaluation | ML and deterministic-baseline evaluation |
| GET | /audit-events | Recent tool decisions and citations |

Complete schemas are available in Swagger at /docs.

## Verification

Run:

    python -m pytest -q
    python tests/acceptance_suite.py

Expected current result:

    24 passed
    PASS: 9 acceptance checks

Build the frontend independently:

    cd frontend
    pnpm build

## Repository structure

    app/                    FastAPI, agent, matching, ML, RAG, and CV extraction
    data/                   Workforce, role, project, and training records
    docs/                   Requirements, RAG, and demo documentation
    frontend/               React/Vite application
    tests/                  Automated and acceptance tests
    docker-compose.yml      Complete PostgreSQL deployment
    generate_data.py        Reproducible data pipeline
    evaluate_system.py      Evaluation report generator
    simulate_hr_review.py   Clearly labelled simulated demo review

## Demo recording guide

The recording is complete. The reusable walkthrough remains available in [the short demo script](docs/demo-video-script.md).

## Evaluation disclosures and known limitations

- Human-labelled matching ground truth is pending. data/matching_ground_truth_template.json is ready for an independent HR reviewer.
- data/simulated_hr_evaluation.json is AI-simulated demonstration evidence and must not be presented as human validation.
- The custom agent is allowed by the proposal, but the RAG integration is custom rather than LangChain or LlamaIndex.
- The UI uses modular custom CSS rather than a named CSS framework.
- Raw /rag/search can return weak semantic neighbors for unrelated text; user-facing /ask applies the scope guard. Threshold calibration is documented in docs/rag-review.md.
- Training links should be verified before production use.

## Additional documentation

- [Requirements compliance matrix](docs/requirements-compliance.md)
- [Hybrid multilingual RAG architecture](docs/rag-review.md)
- [Demo video script](docs/demo-video-script.md)

## Before pushing to GitHub

    git init
    git add .
    git status
    git commit -m "Complete AI talent workforce matching system"

Confirm that .env, local databases, dependency folders, caches, and build output are not listed by git status. Then create a GitHub repository and add its exact remote URL before pushing.
