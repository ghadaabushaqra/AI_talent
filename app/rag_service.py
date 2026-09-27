"""Grounded hybrid retrieval over role, project, and training records.

The primary backend combines multilingual dense E5 embeddings with exact
lexical TF-IDF scores.  A deterministic LSA + lexical backend is retained only
as a labelled recovery path when the dense model cannot be loaded.
"""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


DEFAULT_MODEL = "intfloat/multilingual-e5-small"

ARABIC_QUERY_EXPANSIONS = {
    "مهندس بيانات أول": "Senior Data Engineer",
    "مهندس بيانات كبير": "Senior Data Engineer",
    "مهندس بيانات خبير": "Senior Data Engineer",
    "مهندس بيانات": "Data Engineer",
    "محلل بيانات": "Data Analyst",
    "مهندس تعلم آلي": "Machine Learning Engineer",
    "المهارات المطلوبة": "required skills",
    "مهارات مطلوبة": "required skills",
    "دورة": "training course",
    "تدريب": "training course",
    "دوكر": "Docker",
    "بايثون": "Python",
}


class HybridRAG:
    def __init__(self, documents: list[dict]):
        if not documents:
            raise ValueError("HybridRAG requires at least one source document")
        self.documents = documents
        self.model_name = os.getenv("RAG_MODEL", DEFAULT_MODEL)
        self.dense_weight = min(1.0, max(0.0, float(os.getenv("RAG_DENSE_WEIGHT", "0.75"))))
        self.lexical_weight = 1.0 - self.dense_weight
        self.minimum_score = min(1.0, max(0.0, float(os.getenv("RAG_MIN_SCORE", "0.18"))))
        self.dense_enabled = os.getenv("RAG_DENSE_ENABLED", "true").lower() not in {"0", "false", "no"}
        self.local_files_only = os.getenv("RAG_LOCAL_FILES_ONLY", "true").lower() not in {"0", "false", "no"}
        self.lexical_vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.lexical_matrix = normalize(
            self.lexical_vectorizer.fit_transform([item["text"] for item in documents])
        ).astype("float32")

        self.dense_model = None
        self.dense_matrix: np.ndarray | None = None
        self.dense_index = None
        self.faiss = None
        self.fallback_reason: str | None = None
        self.embedding_dimensions = 0
        self.backend = "initializing"

        if self.dense_enabled:
            self._initialize_dense_backend()
        if self.dense_matrix is None:
            self._initialize_lsa_fallback()

    def _initialize_dense_backend(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self.dense_model = SentenceTransformer(
                self.model_name,
                trust_remote_code=False,
                local_files_only=self.local_files_only,
            )
            passages = [f"passage: {item['text']}" for item in self.documents]
            self.dense_matrix = self.dense_model.encode(
                passages,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype("float32")
            self.embedding_dimensions = int(self.dense_matrix.shape[1])
            try:
                import faiss

                self.faiss = faiss
                self.dense_index = faiss.IndexFlatIP(self.embedding_dimensions)
                self.dense_index.add(self.dense_matrix)
                self.backend = "hybrid_multilingual_e5_faiss"
            except ImportError:
                self.backend = "hybrid_multilingual_e5_numpy"
        except Exception as exc:  # the recovery path is deliberately visible in status metadata
            self.dense_model = None
            self.dense_matrix = None
            self.dense_index = None
            message = " ".join(str(exc).split())[:180]
            self.fallback_reason = f"{type(exc).__name__}: {message}"

    def _initialize_lsa_fallback(self) -> None:
        dimensions = max(
            2,
            min(
                64,
                self.lexical_matrix.shape[0] - 1,
                self.lexical_matrix.shape[1] - 1,
            ),
        )
        self.lsa_model = TruncatedSVD(n_components=dimensions, random_state=7)
        self.lsa_matrix = normalize(self.lsa_model.fit_transform(self.lexical_matrix)).astype("float32")
        self.embedding_dimensions = dimensions
        self.backend = "lsa_lexical_fallback"
        if not self.fallback_reason:
            self.fallback_reason = "Dense embeddings disabled by configuration"

    def status(self) -> dict:
        return {
            "backend": self.backend,
            "dense_available": self.dense_matrix is not None,
            "model": self.model_name if self.dense_matrix is not None else None,
            "embedding_dimensions": self.embedding_dimensions,
            "dense_weight": self.dense_weight if self.dense_matrix is not None else 0.0,
            "lexical_weight": self.lexical_weight if self.dense_matrix is not None else 0.3,
            "minimum_score": self.minimum_score if self.dense_matrix is not None else 0.0,
            "fallback_reason": self.fallback_reason,
            "faiss": self.dense_index is not None,
            "documents": len(self.documents),
        }

    def _lexical_scores(self, query: str) -> np.ndarray:
        vector = normalize(self.lexical_vectorizer.transform([query])).astype("float32")
        scores = self.lexical_matrix @ vector.T
        return scores.toarray().ravel() if hasattr(scores, "toarray") else np.asarray(scores).ravel()

    @staticmethod
    def _expand_query(query: str) -> str:
        additions = [english for arabic, english in ARABIC_QUERY_EXPANSIONS.items() if arabic in query]
        return f"{query} {' '.join(dict.fromkeys(additions))}".strip()

    def _semantic_scores(self, query: str) -> np.ndarray:
        if self.dense_matrix is not None and self.dense_model is not None:
            vector = self.dense_model.encode(
                [f"query: {query}"],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype("float32")
            if self.dense_index is not None:
                values, indices = self.dense_index.search(vector, len(self.documents))
                scores = np.zeros(len(self.documents), dtype="float32")
                for index, value in zip(indices[0], values[0]):
                    if index >= 0:
                        scores[int(index)] = value
                return scores
            return self.dense_matrix @ vector[0]

        vector = normalize(
            self.lsa_model.transform(self.lexical_vectorizer.transform([query]))
        ).astype("float32")
        return self.lsa_matrix @ vector[0]

    def search(
        self,
        query: str,
        limit: int = 3,
        source_types: set[str] | None = None,
    ) -> dict:
        clean_query = " ".join((query or "").split())
        if not clean_query:
            return self._response([])

        expanded_query = self._expand_query(clean_query)
        lexical_scores = np.clip(self._lexical_scores(expanded_query), 0.0, 1.0)
        semantic_scores = np.clip(self._semantic_scores(expanded_query), 0.0, 1.0)
        if self.dense_matrix is not None:
            hybrid_scores = self.dense_weight * semantic_scores + self.lexical_weight * lexical_scores
            minimum_score = self.minimum_score
        else:
            hybrid_scores = 0.7 * semantic_scores + 0.3 * lexical_scores
            minimum_score = 0.0

        allowed: Iterable[int] = range(len(self.documents))
        if source_types:
            allowed = [
                index
                for index, item in enumerate(self.documents)
                if item["source_type"] in source_types
            ]
        ranked = sorted(allowed, key=lambda index: float(hybrid_scores[index]), reverse=True)

        results = []
        for index in ranked:
            hybrid_score = float(hybrid_scores[index])
            if hybrid_score <= minimum_score:
                continue
            item = self.documents[index]
            results.append(
                {
                    **item,
                    "score": round(hybrid_score, 3),
                    "hybrid_score": round(hybrid_score, 3),
                    "dense_score": round(float(semantic_scores[index]), 3)
                    if self.dense_matrix is not None
                    else None,
                    "lexical_score": round(float(lexical_scores[index]), 3),
                    "evidence": item["text"],
                }
            )
            if len(results) >= max(1, min(10, limit)):
                break
        return self._response(results)

    def _response(self, results: list[dict]) -> dict:
        trace = [
            "encode_multilingual_query" if self.dense_matrix is not None else "encode_lsa_query",
            "search_dense_faiss" if self.dense_index is not None else "search_semantic_vectors",
            "search_lexical_tfidf",
            "hybrid_score_fusion",
            "retrieve_organizational_knowledge",
        ]
        return {
            "results": results,
            "retrieval_method": self.backend,
            "embedding_model": self.model_name if self.dense_matrix is not None else None,
            "embedding_dimensions": self.embedding_dimensions,
            "dense_available": self.dense_matrix is not None,
            "fallback_reason": self.fallback_reason,
            "weights": {
                "dense": self.dense_weight if self.dense_matrix is not None else 0.0,
                "lexical": self.lexical_weight if self.dense_matrix is not None else 1.0,
            },
            "tool_trace": trace,
            "citations": [item["id"] for item in results],
        }
