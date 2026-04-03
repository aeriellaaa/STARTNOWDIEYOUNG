from __future__ import annotations
import json, os
from typing import Any, List, Dict
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

SERPAPI_URL = "https://serpapi.com/search"
EMBED_MODEL_NAME = "mixedbread-ai/mxbai-embed-large-v1"
_MODEL = None

def _get_serpapi_key():
    load_dotenv()
    key = os.getenv("SERPAPI_KEY", "").strip()
    if not key:
        raise RuntimeError("SERPAPI_KEY not set")
    return key

def _get_embedder():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    _MODEL = SentenceTransformer(EMBED_MODEL_NAME)
    return _MODEL

def search_patents(keyword, user_description, limit=10):
    keyword = (keyword or "").strip()
    user_description = (user_description or "").strip()
    if not keyword: raise ValueError("keyword required")
    if not user_description: raise ValueError("description required")
    params = {"engine": "google_patents", "q": keyword, "api_key": _get_serpapi_key()}
    resp = requests.get(SERPAPI_URL, params=params, timeout=30)
    resp.raise_for_status()
    organic = resp.json().get("organic_results") or []
    candidates = [{"patent_number": str(r.get("publication_number","")), "title": str(r.get("title","")), "abstract": str(r.get("snippet",""))} for r in organic[:30]]
    if not candidates: return []
    embedder = _get_embedder()
    texts = [user_description] + [c["abstract"] or c["title"] for c in candidates]
    vecs = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    scores = (vecs[1:] @ vecs[0]).tolist()
    for c, s in zip(candidates, scores):
        c["similarity_score"] = float(s)
    candidates.sort(key=lambda x: x["similarity_score"], reverse=True)
    return candidates[:limit]

if __name__ == "__main__":
    results = search_patents("solar energy battery", "solar powered battery storage system for homes")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['patent_number']} - {r['title']} ({r['similarity_score']:.4f})")
