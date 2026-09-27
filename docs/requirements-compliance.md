# Requirements Compliance Matrix

This matrix compares the current implementation with `AI_Talent_Workforce_Matching_Agent (1).docx`. It distinguishes functional compliance from literal technology-stack compliance so that the final report does not overstate what was built.

## Status definitions

- **PASS**: implemented and verified.
- **PASS WITH NOTE**: the requirement works, with a disclosed constraint or development fallback.
- **DEVIATION**: the outcome is partly available, but the implementation differs from the named stack.
- **PENDING HUMAN**: cannot be completed honestly without a human reviewer.
- **MISSING**: not present in the current workspace.

## Compliance matrix

| Area | Document requirement | Status | Implementation evidence | Verification |
|---|---|---:|---|---|
| Source data | Load and clean the supplied employee and resume datasets | PASS | `generate_data.py`, `UpdatedResumeDataSet.csv`, `Employee Sample Data - A.xlsx` | `/health`: 1,000 employees and 962 candidates |
| Employee enrichment | Extend employees with skills, availability and past projects | PASS | `generate_data.py`, `data/employees.json` | Team-builder and directory tests use the generated fields |
| Authored data | Provide approximately 20 projects and 25 training entries | PASS | `data/projects.json`, `data/training_catalog.json` | `/health`: 20 projects and 25 courses |
| CV processing | Extract skills, experience, education, projects and certifications | PASS | `app/cv_extraction.py`, `data/candidates.json` | `test_cv_extraction_preserves_structured_evidence_without_invention` |
| Source evidence | Preserve resume evidence for extracted claims | PASS | Candidate structured fields contain evidence snippets and field status | CV extraction and uploaded-CV tests |
| Unknown handling | Do not convert unavailable evidence into a zero or fabricate it | PASS | `unknown`, `confirmed_absent` and `not_assessed` states in `app/cv_extraction.py` and `app/candidate_matching.py` | `test_unknown_candidate_availability_is_not_scored_as_zero` |
| CV upload | Accept and structure PDF, DOCX and TXT CVs | PASS | `POST /upload-cv` in `app/main.py` | Uploaded profiles are saved and become matchable |
| Requirement extraction | Parse role, required/preferred skills, seniority and requested count | PASS | `extract_requirements()` and dialogue validation | Alias, count and clarification tests |
| Candidate matching | Rank candidates or employees from a role brief | PASS | `POST /match`, `app/candidate_matching.py` | Exact requested result count is verified |
| Weighted baseline | Preserve 40/25/15/10/10 skill, experience, project, availability and qualification weights | PASS | Component weights in `app/candidate_matching.py` | Matching tests inspect structured component scores |
| Explainability | Show matched skills, missing skills, experience, component scores and rationale | PASS | `/match` response and React match cards | Grounding test checks returned record IDs |
| ML probability | Train a supervised logistic-regression success classifier and keep it separate from the weighted score | PASS | `app/ml_evaluation.py`, `app/candidate_matching.py`, `GET /evaluation` | Current deployed evaluation reports both models separately |
| ML comparison | Report classifier performance against the baseline even when the classifier does not win | PASS WITH NOTE | `GET /evaluation` includes accuracy, precision, recall, F1 and confusion matrices | Deployed result: classifier accuracy 0.934; baseline accuracy 0.938 |
| Team builder | Build a complementary team rather than independently selecting the top employees | PASS | `app/workforce_tools.py`, `POST /team-builder` | Tests verify exact team size, coverage and unique contributions |
| Project availability | Check availability across the requested start date and duration | PASS | Allocation-window logic in `app/workforce_tools.py` | Full-period availability test |
| Similar projects | Use relevant past-project evidence in team construction | PASS | `find_similar_projects()` and project-relevance scoring | Tool trace and project IDs are returned |
| Team gap analysis | Return coverage, weak/missing skills and training recommendations | PASS | `/team-builder` response and React Team Builder | Acceptance suite validates the workflow |
| Individual skill gap | Compare an employee with a target role and recommend catalog training | PASS | `POST /skill-gap` | Dialogue and endpoint tests |
| Agent orchestration | Route natural-language requests to matching, team, gap and training tools | PASS | `app/agent_dialogue.py`, `app/ollama_cloud.py`, `POST /ask` | Clarification and tool-routing tests |
| Conversation context | Ask for missing fields and support follow-up requests | PASS | Persisted `ChatTurn` state in SQL | Follow-up availability and multi-turn tests |
| Grounding | Answer from records/tools, cite IDs, reject unsupported scope and disclose weak matches | PASS | Record citations, knowledge citations, scope guard and no-strong-match flags | Weather, weak-match and citation tests |
| Audit trail | Log tool inputs, decisions and cited IDs | PASS | `AuditEvent`, `GET /audit-events` | Dashboard reads the audit trail |
| RAG records | Retrieve traceable role, project and training knowledge | PASS | `POST /rag/search`, knowledge citations in `/ask` | English role and course sanity queries return expected IDs |
| RAG named stack | Use model-generated embeddings with FAISS and LangChain or LlamaIndex | PASS WITH NOTE | Multilingual E5 dense embeddings + TF-IDF hybrid fusion + FAISS are implemented; the retrieval layer remains custom rather than LangChain/LlamaIndex | Arabic query resolves to `ROLE-001`; hybrid scores and source IDs are tested |
| Backend/API | FastAPI endpoints `/health`, `/candidates`, `/match`, `/team-builder`, `/skill-gap`, `/ask` with OpenAPI | PASS | `app/main.py` | 24 automated API/unit tests pass |
| Database | Use PostgreSQL as the reference relational deployment | PASS WITH NOTE | `docker-compose.yml` configures PostgreSQL; direct local execution has a documented SQLite fallback | Running `/health` reports `postgresql`, connected |
| Frontend | Provide a functional React/Vite UI for the documented workflows | PASS | `frontend/src`, `frontend/vite.config.js` | Production build and manual workflow checks passed |
| CSS stack | Use a modern CSS framework | DEVIATION | The React UI uses modular custom CSS rather than Tailwind, Bootstrap or similar | Functional and visual, but not a literal stack match |
| Acceptance questions | Pass documented matching, follow-up, aggregation, team, no-match and out-of-scope scenarios | PASS | `tests/acceptance_suite.py` | 9 of 9 acceptance checks pass |
| Automated tests | Include repeatable tests for core behavior | PASS | `tests/test_api.py`, `tests/acceptance_suite.py` | 24 of 24 pytest tests and 9 of 9 acceptance checks pass |
| Human ground truth | Obtain independent human labels and calculate ranking metrics against them | PENDING HUMAN | `data/matching_ground_truth_template.json` is ready; simulated review is explicitly disclosed | `/evaluation` reports `pending_human_review` and `human_ground_truth: false` |
| Git deliverable | Supply a Git repository with code, tests and history | PASS WITH NOTE | A clean local Git repository is initialized with ignore rules and normalized line endings | Remote GitHub repository and first push are still pending |

## Honest delivery statement

The functional prototype is complete for the documented workflows and acceptance questions. It is not yet a literal 100% match because the RAG framework is custom rather than LangChain/LlamaIndex, human-labelled ground truth is pending, the GitHub remote/push is pending, and the UI uses custom CSS rather than a named CSS framework.

The ML comparison is complete even though the current classifier does not outperform the deterministic baseline. That result should be reported, not hidden or altered.
