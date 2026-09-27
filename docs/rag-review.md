# Hybrid Multilingual RAG Architecture

## Conclusion

The RAG implementation is now grounded, traceable, multilingual and hybrid. Its primary backend combines local `intfloat/multilingual-e5-small` neural embeddings with TF-IDF lexical matching. Normalized 384-dimensional vectors are searched using `faiss.IndexFlatIP` in Docker, and both retrieval signals are exposed in every result.

The retrieval service remains custom rather than LangChain/LlamaIndex. A labelled LSA + lexical recovery backend is retained only when the dense model cannot be loaded; the API never presents that fallback as neural embeddings.

## Current pipeline

1. Build one grounded passage for every target role, project and training course.
2. Prefix passages with `passage: ` and encode them using multilingual E5.
3. L2-normalize the 384-dimensional vectors and add them to `faiss.IndexFlatIP`.
4. Build a separate normalized TF-IDF word and bigram matrix.
5. Expand a small set of explicit Arabic workforce terms into canonical English role/skill terms.
6. Prefix the expanded query with `query: ` and calculate dense and lexical scores.
7. Fuse the scores using 75% dense semantic retrieval and 25% exact lexical retrieval.
8. Apply source-type filtering and a minimum relevance threshold.
9. Return original IDs, evidence, `dense_score`, `lexical_score` and `hybrid_score`.

The implementation is in `app/main.py`, and `/ask` keeps workforce record citations separate from RAG knowledge citations.

## What is already correct

- Retrieved facts come only from role, project and training records.
- Every result retains a stable source ID and its evidence text.
- FAISS uses inner product over normalized vectors, which is the correct construction for cosine-style ranking.
- The corpus is small, so an exact flat index is appropriate and avoids approximate-search errors.
- The agent does not use retrieved text as authority to invent candidate or employee facts.
- RAG evidence and operational-record citations are returned separately.

## Findings from the running system

| Query | Observed result | Assessment |
|---|---|---|
| `What competencies are required for a Senior Data Engineer?` | `ROLE-001` ranked first at 0.917 | Correct English role retrieval |
| `What course remedies a Docker competency gap?` | `TRN-07`, `TRN-23`, `TRN-05` | Relevant training retrieval |
| `ما المهارات المطلوبة لمهندس بيانات أول؟` | `ROLE-001` ranked first; dense and lexical evidence are both positive | Correct cross-lingual role retrieval |

The dense index reports 384 dimensions from the pretrained E5 model. The fallback index has corpus-derived dimensions and is explicitly labelled `lsa_lexical_fallback`.

## Remaining gaps and constraints

1. **Framework mismatch.** There is no LangChain or LlamaIndex integration, even though the proposal lists that layer.
2. **Startup encoding.** The small corpus is re-encoded at application startup; persisted index versioning is still a future optimization.
3. **Model footprint.** E5 and its CPU runtime make the API Docker image materially larger than the former LSA-only implementation.
4. **Threshold calibration.** The current 0.18 hybrid threshold is conservative and should eventually be calibrated with a broader bilingual retrieval set.
5. **Fallback limitations.** LSA recovery preserves English keyword retrieval but cannot provide the same cross-lingual behavior.

## Implemented free architecture

### Embedding model

Use `intfloat/multilingual-e5-small`:

- MIT-licensed and free to run locally.
- 384-dimensional dense vectors.
- Designed for information retrieval rather than only generic sentence similarity.
- Multilingual, including cross-lingual retrieval suitable for Arabic queries over English records.
- Requires `query: ` before queries and `passage: ` before indexed records, including non-English text.

Model card: <https://huggingface.co/intfloat/multilingual-e5-small>

### Vector store

Continue using FAISS `IndexFlatIP`:

- Encode passages and queries as `float32`.
- L2-normalize both before adding/searching.
- Use inner product so the score is cosine similarity.
- Keep the exact flat index because the current corpus is only dozens of records.

FAISS guidance: <https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances>

### Framework decision

For the cleanest implementation, place model loading, encoding, persistence and search behind a small `EmbeddingProvider` interface and keep the existing FastAPI response contract.

If literal stack compliance is required by the evaluator, connect the same E5 model and FAISS index through LlamaIndex. The framework must not be allowed to replace record IDs, evidence fields, source-type filtering or audit logging. Adding LlamaIndex solely as an adapter is preferable to rewriting the working agent.

## Implemented safeguards

1. RAG construction and retrieval live in `app/rag_service.py`.
2. Configuration is available through:
   - `RAG_BACKEND=e5` for the new primary backend.
   - `RAG_MODEL=intfloat/multilingual-e5-small`.
   - `RAG_ALLOW_LSA_FALLBACK=true` for development recovery only.
3. Indexed text and queries use the E5 `passage:` and `query:` prefixes.
4. Dense vectors are normalized and searched through `faiss.IndexFlatIP(384)` in Docker.
5. Existing response keys and grounded evidence are preserved.
6. `/health` exposes the active backend, model, dimensions, weights and fallback reason.
7. Dense-loading failures activate `lsa_lexical_fallback`; that response never claims neural embeddings.
8. Candidate ranking, ML probability and team optimization remain isolated from RAG scoring.

## Validation contract

- English role query returns `ROLE-001` in the top three.
- English Docker training query returns a Docker course in the top three.
- Arabic Senior Data Engineer query returns `ROLE-001` in the top three.
- Source-type filtering never leaks courses into role-only retrieval or vice versa.
- Every returned source ID exists in the indexed metadata and retains exact evidence text.
- The user-facing `/ask` endpoint rejects unrelated and out-of-scope queries through its scope guard. Raw `/rag/search` threshold calibration remains a documented limitation and can return weak semantic neighbors.
- Reloading a persisted index produces identical top-K ordering for the fixed evaluation set.
- `/ask` continues separating `record_citations` from `knowledge_citations`.
- Candidate ranking, ML probability and deterministic weighted scores remain unchanged; RAG must not affect those calculations.

## Dependency and deployment impact

`sentence-transformers` brings a larger model/runtime footprint than the current scikit-learn pipeline, and the model must be downloaded during image build or mounted from a local cache. There is no per-request API charge after the model is present. Pin the package version and model revision so Docker builds remain reproducible.

The hybrid implementation is isolated to the retrieval layer. Candidate matching, team optimization, SQL records, evidence extraction, agent validation and citations retain their existing contracts.
