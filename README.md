# TalentAI

HR helper for searching people, matching candidates, building teams, and answering workforce questions from the two project datasets.

It does not replace a recruiter. If a CV does not mention something, the app marks it as unknown instead of guessing.

## Demo

[Google Drive video](https://drive.google.com/file/d/1GK46ov3YtgyY8Xc3cXbg5qg46XqeKpQ9/view?usp=sharing)

Walkthrough notes: [docs/demo-video-script.md](docs/demo-video-script.md)

## What it does

- Dashboard with dataset counts and shortcuts
- People directory (employees + candidates), with skill / department / availability filters
- Person profile with skills, projects, availability, and source text
- Candidate matching with a weighted score (40 / 25 / 15 / 10 / 10)
- ML success probability shown next to the score, not mixed into it
- Team builder that looks at skills, schedules, and past projects
- Skill-gap view plus training suggestions from the catalog
- CV upload (PDF, DOCX, TXT)
- Ask AI chat with citations
- Hybrid RAG: multilingual E5 + TF-IDF + FAISS

## Stack

Frontend is React (Vite). API is FastAPI. Postgres runs in Docker; SQLite is only for a quick local run without Compose.

Matching is a normal weighted score. The ML part is a scikit-learn logistic regression and stays separate. RAG uses `intfloat/multilingual-e5-small`. Ollama Cloud is optional; if there is no key the agent still works with the backend tools.

```
React (localhost:5173)
        |
        v
FastAPI  -->  matching / team / skill-gap / CV / RAG / chat
        |
        v
PostgreSQL  (built from the Excel + CSV files)
```

## Run with Docker

Need Docker Desktop (WSL 2) and a bit of RAM, around 6 GB is comfortable.

The app runs without a cloud key. If you want Ollama Cloud, copy `.env.example` to `.env` and put the key there:

```
Copy-Item .env.example .env
```

```
OLLAMA_API_KEY=your_private_key
OLLAMA_CLOUD_MODEL=gpt-oss:20b-cloud
```

Keep `.env` on your machine only.

```
docker compose up --build
```

First start pulls the E5 model, so it can be slow.

- App: http://localhost:5173
- API docs: http://localhost:8765/docs
- Health: http://localhost:8765/health

```
docker compose down
```

## Run without Docker

This path uses SQLite.

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python generate_data.py
python -m uvicorn app.main:app --reload --port 8765
```

Then in another terminal:

```
cd frontend
corepack enable
pnpm install --frozen-lockfile
pnpm dev
```

## Data

| File | What it is |
|---|---|
| `UpdatedResumeDataSet.csv` | Resume text used for candidates |
| `Employee Sample Data - A.xlsx` | Employee spreadsheet |
| `generate_data.py` | Cleans / builds the JSON under `data/` |
| `data/candidates.json` | Candidates + CV evidence |
| `data/employees.json` | Employee profiles |
| `data/projects.json` | Projects (teams + RAG) |
| `data/training_catalog.json` | Training list for skill gaps |
| `data/target_roles.json` | Target roles |

Re-run `python generate_data.py` if you change the source files or the extraction code.

## Matching

| Part | Weight |
|---|---:|
| Skills | 40% |
| Experience | 25% |
| Relevant projects | 15% |
| Availability | 10% |
| Education / certs | 10% |

Missing fields are **Not Assessed**. Those weights get split over the parts we actually have. A confirmed “no” and a missing field are not the same thing. The ML probability does not change the weighted score.

## RAG / chat

Index covers roles, projects, and training. Search mix is about 75% E5 and 25% TF-IDF (FAISS, 384-d).

The chat stays on workforce topics, asks when something is missing, and only runs tools the backend already allows. Employee/candidate citations are separate from RAG citations. If Ollama is on, it can help plan the wording; the server still checks fields, tools, and evidence.

## API

| Method | Path | Notes |
|---|---|---|
| GET | `/health` | DB, datasets, RAG, LLM |
| GET | `/dashboard` | Counts / activity |
| GET | `/people` | Search |
| GET | `/people/{person_id}` | Full profile |
| POST | `/upload-cv` | Parse a CV |
| POST | `/match` | Matching |
| POST | `/team-builder` | Team |
| POST | `/skill-gap` | Gaps + training |
| POST | `/rag/search` | Raw retrieval |
| POST | `/ask` | Chat |
| GET | `/evaluation` | ML vs baseline |
| GET | `/audit-events` | Recent tool use |

Full schemas are in `/docs`.

## Tests

```
python -m pytest -q
python tests/acceptance_suite.py
```

Last run here: 24 pytest tests passed, 9 acceptance checks passed.

```
cd frontend
pnpm build
```

## Folder layout

```
app/                    API, matching, ML, RAG, CV parsing
data/                   generated JSON
docs/                   extra notes
frontend/               React app
tests/
docker-compose.yml
generate_data.py
evaluate_system.py
simulate_hr_review.py   demo review only, not a real HR sign-off
```

## Limits

- No human-labelled matching ground truth yet. Template: `data/matching_ground_truth_template.json`
- `data/simulated_hr_evaluation.json` is a simulated demo file. Do not treat it as a real reviewer result.
- Agent + RAG are custom (not LangChain / LlamaIndex). That matches the proposal, just worth stating.
- UI CSS is custom, not Bootstrap/Tailwind.
- `/rag/search` can still surface weak neighbors; `/ask` has a tighter scope check. See `docs/rag-review.md`
- Training URLs should be double-checked before any real use

More detail: [requirements](docs/requirements-compliance.md), [RAG](docs/rag-review.md).
