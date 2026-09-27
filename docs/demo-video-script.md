# Short Demo Video Script

Recommended duration: **4–6 minutes**.

## Before recording

1. Start Docker Desktop.
2. Run docker compose up --build.
3. Open <http://localhost:5173>.
4. Keep <http://localhost:8765/health> available in another tab.
5. Close terminals or files that display .env or an API key.
6. Use a clean browser window and zoom so labels and evidence remain readable.

## Recording sequence

### 1. Introduction — 20 seconds

Say:

> “TalentAI is an evidence-grounded workforce intelligence system built from the supplied employee and resume datasets. It supports people discovery, explainable matching, team construction, skill-gap analysis, CV processing, and a conversational HR agent.”

Show the dashboard counts and working quick actions.

### 2. People and profiles — 35 seconds

- Open **People Directory**.
- Switch between Employees and Candidates.
- Filter by NLP or AWS.
- Open a profile and point to its record ID and source evidence.
- Open **Person Profile**, search for an employee, and show availability, skills, experience, and linked projects.

### 3. Candidate matching — 60 seconds

Use:

    Senior Data Engineer requiring Python, AWS and SQL

- Select Candidates.
- Request exactly 3 results.
- Point out matched and missing skills, component scores, weighted match, ML probability, explanation, and record citation.
- State that the weighted score and ML probability are separate.

### 4. Team Builder — 60 seconds

Use:

    Build a 4-person team for a six-month NLP project requiring Python, NLP, RAG and AWS.

- Show that exactly four employees are returned when feasible.
- Point out complementary responsibilities, complete-period availability, skill coverage, similar projects, gaps, and training.

### 5. Skill gap and training — 40 seconds

- Select an employee from the searchable list.
- Choose **Senior Data Engineer**.
- Run the analysis.
- Show current skills, missing requirements, and official training links.

### 6. Grounded assistant — 60 seconds

First ask:

    I need a team for an NLP project.

Show that the assistant asks for missing team size, duration, or skills instead of guessing. Complete the follow-up and show:

- tool trace;
- employee record citations;
- separate RAG knowledge citations;
- retrieved role, project, or training evidence.

Then ask:

    What is the weather tomorrow?

Show the scope guard rejecting the unrelated request.

### 7. Technical proof — 30 seconds

Open /health and point out:

- PostgreSQL is connected;
- employee and candidate counts;
- hybrid_multilingual_e5_faiss is active;
- embedding dimensions are 384;
- whether Ollama Cloud is configured.

Finish by briefly showing the GitHub repository structure and README without opening .env.

## Final statement

> “All recommendations are grounded in loaded records and retain source IDs. Missing CV evidence is not fabricated, deterministic matching remains separate from ML probability, and the final decision stays with the HR reviewer.”
