"""
SerpAPI Google Patents keyword search + embedding reranking.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List
from urllib.parse import quote_plus

import requests
from dotenv import load_dotenv

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:  # pragma: no cover
    SentenceTransformer = None  # type: ignore[assignment]
    _SENTENCE_TRANSFORMERS_IMPORT_ERROR = e
else:
    _SENTENCE_TRANSFORMERS_IMPORT_ERROR = None


SERPAPI_URL = "https://serpapi.com/search"
SERPAPI_ENGINE = "google_patents"

EMBED_MODEL_NAME = "mixedbread-ai/mxbai-embed-large-v1"

_MODEL: SentenceTransformer | None = None


def _get_serpapi_key() -> str:
    load_dotenv()
    key = os.getenv("SERPAPI_KEY", "").strip()
    if not key:
        raise RuntimeError("SERPAPI_KEY is not set in the environment or .env file")
    return key


def _download_model_files(cache_dir: str) -> str:
    """
    Download required SentenceTransformer files using plain HTTP.

    This avoids depending on HF cache behavior inside restricted environments.
    """
    local_model_dir = os.path.join(cache_dir, EMBED_MODEL_NAME.replace("/", "-"))
    os.makedirs(local_model_dir, exist_ok=True)

    base_url = f"https://huggingface.co/{EMBED_MODEL_NAME}/resolve/main"

    def download_file(url: str, dest_path: str) -> None:
        tmp_path = dest_path + ".part"
        headers = {"User-Agent": "Mozilla/5.0"}
        with requests.get(url, stream=True, timeout=60, headers=headers) as r:
            r.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        os.replace(tmp_path, dest_path)

    required_files = [
        "config.json",
        "config_sentence_transformers.json",
        "sentence_bert_config.json",
        "modules.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
        "model.safetensors",
        "1_Pooling/config.json",
    ]

    for fn in required_files:
        dest = os.path.join(local_model_dir, fn)
        if os.path.exists(dest):
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        download_file(f"{base_url}/{fn}", dest)

    return local_model_dir


def _get_embedder() -> SentenceTransformer:
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    if SentenceTransformer is None:  # pragma: no cover
        raise RuntimeError(
            "sentence-transformers is required but could not be imported."
        ) from _SENTENCE_TRANSFORMERS_IMPORT_ERROR

    cache_dir = os.path.join(os.path.dirname(__file__), ".hf_cache_sentence_transformers")
    os.makedirs(cache_dir, exist_ok=True)
    local_model_dir = _download_model_files(cache_dir)

    _MODEL = SentenceTransformer(local_model_dir)
    return _MODEL


def _serpapi_search_patents(keyword: str, limit: int) -> List[Dict[str, str]]:
    api_key = _get_serpapi_key()
    params = {"engine": SERPAPI_ENGINE, "q": keyword, "api_key": api_key}

    try:
        resp = requests.get(SERPAPI_URL, params=params, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"SerpAPI request failed: {e}") from e

    try:
        data: Any = resp.json()
    except json.JSONDecodeError as e:
        snippet = (resp.text or "")[:200]
        raise RuntimeError(
            f"SerpAPI returned invalid JSON: {e}. Body starts with: {snippet!r}"
        ) from e

    organic = data.get("organic_results") or []
    if not isinstance(organic, list):
        raise RuntimeError("Unexpected SerpAPI response: 'organic_results' is not a list")

    out: List[Dict[str, str]] = []
    for item in organic:
        if not isinstance(item, dict):
            continue
        out.append(
            {
                "patent_number": str(item.get("publication_number") or "").strip(),
                "title": str(item.get("title") or "").strip(),
                "abstract": str(item.get("snippet") or "").strip(),
            }
        )
        if len(out) >= limit:
            break

    return out


def search_patents(
    keyword: str,
    user_description: str,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """
    Fetch top patents using SerpAPI, rerank by cosine similarity between:
      - embedding(user_description)
      - embedding(patent abstract/snippet)

    Returns top `limit` dicts with:
      patent_number, title, abstract, similarity_score
    """
    keyword = (keyword or "").strip()
    user_description = (user_description or "").strip()
    if not keyword:
        raise ValueError("keyword must be non-empty")
    if not user_description:
        raise ValueError("user_description must be non-empty")
    if limit < 1:
        raise ValueError("limit must be >= 1")

    candidates = _serpapi_search_patents(keyword, limit=max(limit * 3, 30))
    if not candidates:
        return []

    embedder = _get_embedder()
    abstracts = [p.get("abstract", "") for p in candidates]
    texts = [user_description] + abstracts

    vectors = embedder.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    if vectors.ndim != 2 or vectors.shape[1] != 1024:
        raise RuntimeError(f"Expected 1024-d embeddings, got shape {vectors.shape}")

    description_vec = vectors[0]
    abstract_vecs = vectors[1:]
    scores = (abstract_vecs @ description_vec).tolist()

    results: List[Dict[str, Any]] = []
    for p, s in zip(candidates, scores):
        results.append(
            {
                "patent_number": p["patent_number"],
                "title": p["title"],
                "abstract": p["abstract"],
                "similarity_score": float(s),
            }
        )

    results.sort(key=lambda x: x["similarity_score"], reverse=True)
    return results[:limit]


if __name__ == "__main__":
    keyword = "solar energy battery"
    description = "I want to make a solar powered battery storage system for homes"
    top = search_patents(keyword, description, limit=10)
    for i, r in enumerate(top, 1):
        print(f"{i}. {r['patent_number']} — {r['title']} (score={r['similarity_score']:.4f})")

