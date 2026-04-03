"""
Patent search + embedding-based reranking.

Fetches patents from SerpAPI (Google Patents engine), then reranks them by
cosine similarity between:
  - embedding(user_description)
  - embedding(patent abstract/snippet)

No duplicate blocks: this file contains a single clean implementation.
"""

from __future__ import annotations

import json
import os
from typing import Any, List, Dict, Tuple

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

# mixedbread-ai: mxbai-embed-large-v1 (1024-d vectors)
EMBED_MODEL_NAME = "mixedbread-ai/mxbai-embed-large-v1"

_MODEL: SentenceTransformer | None = None


def _get_serpapi_key() -> str:
    load_dotenv()
    key = os.getenv("SERPAPI_KEY", "").strip()
    if not key:
        raise RuntimeError("SERPAPI_KEY is not set in the environment or .env file")
    return key


def _download_required_model_files(cache_dir: str) -> str:
    """
    Download required model files directly via HTTP.

    We download to a local directory and load SentenceTransformer from that
    directory, avoiding any broken/incomplete HF cache snapshots.
    """
    local_model_dir = os.path.join(cache_dir, EMBED_MODEL_NAME.replace("/", "-"))
    os.makedirs(local_model_dir, exist_ok=True)

    base_url = f"https://huggingface.co/{EMBED_MODEL_NAME}/resolve/main"

    def _download_file(url: str, dest_path: str) -> None:
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
        # Pooling submodule config is required (contains word_embedding_dimension)
        "1_Pooling/config.json",
    ]

    for fn in required_files:
        dest = os.path.join(local_model_dir, fn)
        if os.path.exists(dest):
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        _download_file(f"{base_url}/{fn}", dest)

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
    local_model_dir = _download_required_model_files(cache_dir)

    _MODEL = SentenceTransformer(local_model_dir)
    return _MODEL


def _serpapi_search_patents(keyword: str, *, limit: int) -> List[Dict[str, str]]:
    api_key = _get_serpapi_key()
    params = {"engine": SERPAPI_ENGINE, "q": keyword, "api_key": api_key}

    resp = requests.get(SERPAPI_URL, params=params, timeout=30)
    resp.raise_for_status()

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
                # Use snippet as abstract text for embedding.
                "abstract": str(item.get("snippet") or "").strip(),
            }
        )
        if len(out) >= limit:
            break

    return out


def _rerank_by_embedding(
    user_description: str,
    patents: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    embedder = _get_embedder()

    abstracts = [(p.get("abstract") or p.get("title") or "") for p in patents]
    texts = [user_description] + abstracts

    vectors = embedder.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,  # cosine similarity == dot product
    )

    # mxbai-embed-large-v1 should be 1024-d
    if vectors.ndim != 2 or vectors.shape[1] != 1024:
        raise RuntimeError(f"Expected 1024-d embeddings, got shape {vectors.shape}")

    description_vec = vectors[0]
    abstract_vecs = vectors[1:]

    # Dot product due to normalization.
    scores = (abstract_vecs @ description_vec).tolist()

    reranked: List[Dict[str, Any]] = []
    for p, s in zip(patents, scores):
        p2 = dict(p)
        p2["similarity_score"] = float(s)
        reranked.append(p2)

    reranked.sort(key=lambda x: float(x.get("similarity_score", 0.0)), reverse=True)
    return reranked


def search_patents(keyword: str, user_description: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Search patents with SerpAPI and rerank using embedding similarity.

    Returns top `limit` dicts with:
      - patent_number (from publication_number)
      - title
      - abstract (SerpAPI snippet)
      - similarity_score
    """
    keyword = (keyword or "").strip()
    user_description = (user_description or "").strip()

    if not keyword:
        raise ValueError("keyword must be non-empty")
    if not user_description:
        raise ValueError("user_description must be non-empty")
    if limit < 1:
        raise ValueError("limit must be >= 1")

    # Pull more candidates so reranking can reorder top N.
    candidate_count = max(limit, min(30, limit * 3))
    candidates = _serpapi_search_patents(keyword, limit=candidate_count)
    if not candidates:
        return []

    reranked = _rerank_by_embedding(user_description, candidates)
    return reranked[:limit]


if __name__ == "__main__":
    keyword = "solar energy battery"
    description = "I want to make a solar powered battery storage system for homes"
    results = search_patents(keyword, description, limit=10)

    print(f"Top {len(results)} reranked patents for {keyword!r}:\n")
    for i, p in enumerate(results, 1):
        print(f"{i}. {p['patent_number']} — {p['title']}")
        print(f"   similarity_score: {p['similarity_score']:.4f}")
        preview = p.get("abstract", "")
        preview = (preview[:400] + "…") if len(preview) > 400 else preview
        print(f"   {preview}\n")

"""
Patent keyword search + embedding-based reranking.

Fetches results via SerpAPI (Google Patents engine), then reranks by cosine
similarity between the user description and each patent abstract/snippet
using sentence-transformers.
"""

# from __future__ import annotations  (duplicate block appended to file)

import json
import os
import shutil
from typing import Any

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

# Mixedbread model: expected 1024-d embeddings.
EMBED_MODEL_NAME = "mixedbread-ai/mxbai-embed-large-v1"

_MODEL: SentenceTransformer | None = None


def _get_serpapi_key() -> str:
    load_dotenv()
    key = os.getenv("SERPAPI_KEY", "").strip()
    if not key:
        raise RuntimeError("SERPAPI_KEY is not set in the environment or .env file")
    return key


def _get_embedder() -> SentenceTransformer:
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    if SentenceTransformer is None:  # pragma: no cover
        raise RuntimeError(
            "sentence-transformers is required but could not be imported."
        ) from _SENTENCE_TRANSFORMERS_IMPORT_ERROR

    # We download model files directly with `requests` because `huggingface_hub`
    # downloads are leaving an empty/incomplete cache snapshot in this environment.
    cache_dir = os.path.join(os.path.dirname(__file__), ".hf_cache_sentence_transformers")
    os.makedirs(cache_dir, exist_ok=True)
    local_model_dir = os.path.join(cache_dir, "mixedbread-ai-mxbai-embed-large-v1")
    os.makedirs(local_model_dir, exist_ok=True)

    def _download_file(url: str, dest_path: str) -> None:
        tmp_path = dest_path + ".part"
        headers = {"User-Agent": "Mozilla/5.0"}
        with requests.get(url, stream=True, timeout=60, headers=headers) as r:
            r.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        os.replace(tmp_path, dest_path)

    base_url = f"https://huggingface.co/{EMBED_MODEL_NAME}/resolve/main"
    required_files = [
        "config.json",
        "config_sentence_transformers.json",
        "sentence_bert_config.json",
        "modules.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
        "1_Pooling/config.json",
        "model.safetensors",
    ]

    for fn in required_files:
        dest = os.path.join(local_model_dir, fn)
        if not os.path.exists(dest):
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            _download_file(f"{base_url}/{fn}", dest)

    _MODEL = SentenceTransformer(local_model_dir)
    return _MODEL


def _serpapi_search_patents(
    keyword: str, *, limit: int, timeout: float
) -> list[dict[str, str]]:
    api_key = _get_serpapi_key()

    params = {
        "engine": SERPAPI_ENGINE,
        "q": keyword,
        "api_key": api_key,
    }

    try:
        resp = requests.get(SERPAPI_URL, params=params, timeout=timeout)
    except requests.RequestException as e:
        raise RuntimeError(f"SerpAPI request failed: {e}") from e

    if not resp.ok:
        snippet = (resp.text or "")[:800].strip()
        raise RuntimeError(
            f"SerpAPI HTTP {resp.status_code}: {resp.reason}. {snippet}"
        )

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

    results: list[dict[str, str]] = []
    for item in organic:
        if not isinstance(item, dict):
            continue
        results.append(
            {
                "patent_number": str(item.get("publication_number") or "").strip(),
                "title": str(item.get("title") or "").strip(),
                # SerpAPI provides snippet text; we use it as abstract for embedding.
                "abstract": str(item.get("snippet") or "").strip(),
            }
        )
        if len(results) >= limit:
            break

    return results


def search_patents(
    keyword: str,
    user_description: str,
    *,
    limit: int = 10,
    timeout: float = 30.0,
    max_patents_for_rerank: int = 30,
) -> list[dict[str, Any]]:
    """
    Returns top `limit` patents reranked by cosine similarity between:
      - embedding(user_description)
      - embedding(patent abstract/snippet)
    """
    keyword = keyword.strip()
    if not keyword:
        raise ValueError("keyword must be non-empty")

    user_description = (user_description or "").strip()
    if not user_description:
        raise ValueError("user_description must be non-empty")

    if limit < 1:
        raise ValueError("limit must be >= 1")

    # Fetch extra candidates so reranking can reorder.
    initial_limit = max(limit, min(max_patents_for_rerank, 100))
    candidates = _serpapi_search_patents(
        keyword, limit=initial_limit, timeout=timeout
    )
    if not candidates:
        return []

    embedder = _get_embedder()
    abstracts = [(c.get("abstract") or c.get("title") or "") for c in candidates]

    # Batch encode: first vector is description, rest are abstracts/snippets.
    vectors = embedder.encode(
        [user_description] + abstracts,
        convert_to_numpy=True,
        normalize_embeddings=True,  # cosine == dot product
    )

    if vectors.ndim != 2 or vectors.shape[1] != 1024:
        raise RuntimeError(f"Expected 1024-d embeddings, got shape {vectors.shape}")

    description_vec = vectors[0]
    abstract_vecs = vectors[1:]
    scores = (abstract_vecs @ description_vec).tolist()  # dot == cosine

    for cand, score in zip(candidates, scores):
        cand["similarity_score"] = float(score)

    candidates.sort(
        key=lambda x: float(x.get("similarity_score", 0.0)), reverse=True
    )
    return candidates[:limit]


# Preserve the embedding/reranking version even if legacy duplicated blocks
# later in the file redefine `search_patents`.
_SEARCH_PATENTS_EMBEDDING = search_patents


if __name__ == "__main__":
    keyword = "solar energy battery"
    description = "I want to make a solar powered battery storage system for homes"
    results = search_patents(keyword, description, limit=10)
    print(f"Top {len(results)} reranked patents for {keyword!r}:\n")
    for i, p in enumerate(results, 1):
        print(f"{i}. {p['patent_number']} — {p['title']}")
        print(f"   similarity_score: {p['similarity_score']:.4f}")
        abstract = p.get("abstract", "")
        preview = (abstract[:400] + "…") if len(abstract) > 400 else abstract
        print(f"   {preview}\n")
    raise SystemExit(0)

"""
Patent keyword search + embedding-based reranking.

Fetches results via SerpAPI (Google Patents engine), then reranks by cosine
similarity between the user description and each patent abstract/snippet
using sentence-transformers.
"""

# from __future__ import annotations  # duplicate block appended to file (leave as no-op)

import json
import os
from typing import Any
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

# Mixedbread model (1024-d output).
EMBED_MODEL_NAME = "mixedbread-ai/mxbai-embed-large-v1"

_MODEL: SentenceTransformer | None = None


def _get_serpapi_key() -> str:
    load_dotenv()
    key = os.getenv("SERPAPI_KEY", "").strip()
    if not key:
        raise RuntimeError("SERPAPI_KEY is not set in the environment or .env file")
    return key


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
    _MODEL = SentenceTransformer(EMBED_MODEL_NAME, cache_folder=cache_dir)
    return _MODEL


def _serpapi_search_patents(
    keyword: str, *, limit: int, timeout: float
) -> list[dict[str, str]]:
    api_key = _get_serpapi_key()

    params = {
        "engine": SERPAPI_ENGINE,
        "q": keyword,
        "api_key": api_key,
    }

    try:
        resp = requests.get(SERPAPI_URL, params=params, timeout=timeout)
    except requests.RequestException as e:
        raise RuntimeError(f"SerpAPI request failed: {e}") from e

    if not resp.ok:
        snippet = (resp.text or "")[:800].strip()
        raise RuntimeError(f"SerpAPI HTTP {resp.status_code}: {resp.reason}. {snippet}")

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

    results: list[dict[str, str]] = []
    for item in organic:
        if not isinstance(item, dict):
            continue
        results.append(
            {
                "patent_number": str(item.get("publication_number") or "").strip(),
                "title": str(item.get("title") or "").strip(),
                # SerpAPI provides snippet text; we use it as abstract for embedding.
                "abstract": str(item.get("snippet") or "").strip(),
            }
        )
        if len(results) >= limit:
            break

    return results


def search_patents(
    keyword: str,
    user_description: str,
    *,
    limit: int = 10,
    timeout: float = 30.0,
    max_patents_for_rerank: int = 30,
) -> list[dict[str, Any]]:
    """
    Returns top `limit` patents reranked by cosine similarity between:
      - embedding(user_description)
      - embedding(patent abstract/snippet)
    """
    keyword = keyword.strip()
    if not keyword:
        raise ValueError("keyword must be non-empty")

    user_description = (user_description or "").strip()
    if not user_description:
        raise ValueError("user_description must be non-empty")

    if limit < 1:
        raise ValueError("limit must be >= 1")

    # Fetch extra candidates so reranking can reorder.
    initial_limit = max(limit, min(max_patents_for_rerank, 100))
    candidates = _serpapi_search_patents(keyword, limit=initial_limit, timeout=timeout)
    if not candidates:
        return []

    embedder = _get_embedder()
    abstracts = [
        (c.get("abstract") or c.get("title") or "") for c in candidates
    ]

    texts = [user_description] + abstracts
    vectors = embedder.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,  # cosine == dot product
    )

    if vectors.ndim != 2 or vectors.shape[1] != 1024:
        raise RuntimeError(f"Expected 1024-d embeddings, got shape {vectors.shape}")

    description_vec = vectors[0]
    abstract_vecs = vectors[1:]
    scores = (abstract_vecs @ description_vec).tolist()

    for cand, score in zip(candidates, scores):
        cand["similarity_score"] = float(score)

    candidates.sort(key=lambda x: float(x.get("similarity_score", 0.0)), reverse=True)
    return candidates[:limit]


if __name__ == "__main__":
    keyword = "solar energy battery"
    description = "I want to make a solar powered battery storage system for homes"
    results = search_patents(keyword, description, limit=10)
    print(f"Top {len(results)} reranked patents for {keyword!r}:\n")
    for i, p in enumerate(results, 1):
        print(f"{i}. {p['patent_number']} — {p['title']}")
        print(f"   similarity_score: {p['similarity_score']:.4f}")
        abstract = p.get("abstract", "")
        preview = (abstract[:400] + "…") if len(abstract) > 400 else abstract
        print(f"   {preview}\n")

"""
Patent keyword search + embedding-based reranking using SerpAPI + sentence-transformers.
"""

# from __future__ import annotations  (duplicate legacy block; commented out)

import json
import os
from typing import Any

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
EMBED_MODEL_NAME = "mxbai-embed-large-v1"

_MODEL: SentenceTransformer | None = None


def _get_serpapi_key() -> str:
    load_dotenv()
    key = os.getenv("SERPAPI_KEY", "").strip()
    if not key:
        raise RuntimeError("SERPAPI_KEY is not set in the environment or .env file")
    return key


def _get_embedder() -> SentenceTransformer:
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    if SentenceTransformer is None:  # pragma: no cover
        raise RuntimeError(
            "sentence-transformers is required but could not be imported."
        ) from _SENTENCE_TRANSFORMERS_IMPORT_ERROR

    _MODEL = SentenceTransformer(EMBED_MODEL_NAME)
    return _MODEL


def _serpapi_search_patents(
    keyword: str, *, limit: int, timeout: float
) -> list[dict[str, str]]:
    api_key = _get_serpapi_key()
    params = {
        "engine": SERPAPI_ENGINE,
        "q": keyword,
        "api_key": api_key,
    }

    try:
        resp = requests.get(SERPAPI_URL, params=params, timeout=timeout)
    except requests.RequestException as e:
        raise RuntimeError(f"SerpAPI request failed: {e}") from e

    if not resp.ok:
        snippet = (resp.text or "")[:800].strip()
        raise RuntimeError(f"SerpAPI HTTP {resp.status_code}: {resp.reason}. {snippet}")

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

    results: list[dict[str, str]] = []
    for item in organic:
        if not isinstance(item, dict):
            continue
        results.append(
            {
                "patent_number": str(item.get("publication_number") or "").strip(),
                "title": str(item.get("title") or "").strip(),
                # SerpAPI provides a snippet; we use it as the "abstract" text to embed.
                "abstract": str(item.get("snippet") or "").strip(),
            }
        )
        if len(results) >= limit:
            break

    return results


def search_patents(
    keyword: str,
    user_description: str,
    *,
    limit: int = 10,
    timeout: float = 30.0,
    max_patents_for_rerank: int = 30,
) -> list[dict[str, Any]]:
    """
    Returns top `limit` patents reranked by cosine similarity between:
      - embedding(user_description)
      - embedding(patent abstract/snippet)
    """
    keyword = keyword.strip()
    if not keyword:
        raise ValueError("keyword must be non-empty")

    user_description = (user_description or "").strip()
    if not user_description:
        raise ValueError("user_description must be non-empty")

    if limit < 1:
        raise ValueError("limit must be >= 1")

    # Fetch extra candidates so reranking has a chance to reorder the top N.
    initial_limit = max(limit, min(max_patents_for_rerank, 100))
    candidates = _serpapi_search_patents(keyword, limit=initial_limit, timeout=timeout)
    if not candidates:
        return []

    embedder = _get_embedder()

    abstracts = [
        c.get("abstract", "") or c.get("title", "") or "" for c in candidates
    ]

    texts = [user_description] + abstracts
    vectors = embedder.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,  # cosine similarity == dot product
    )

    # mxbai-embed-large-v1 is expected to produce 1024-d vectors.
    if vectors.ndim != 2 or vectors.shape[1] != 1024:
        raise RuntimeError(f"Expected 1024-d embeddings, got shape {vectors.shape}")

    description_vec = vectors[0]
    abstract_vecs = vectors[1:]

    # Dot product due to normalization.
    scores = (abstract_vecs @ description_vec).tolist()

    for cand, score in zip(candidates, scores):
        cand["similarity_score"] = float(score)

    candidates.sort(key=lambda x: float(x.get("similarity_score", 0.0)), reverse=True)
    return candidates[:limit]


if __name__ == "__main__":
    keyword = "solar energy battery"
    description = "I want to make a solar powered battery storage system for homes"
    results = search_patents(keyword, description, limit=10)
    print(f"Top {len(results)} reranked patents for {keyword!r}:\n")
    for i, p in enumerate(results, 1):
        print(f"{i}. {p['patent_number']} — {p['title']}")
        print(f"   similarity_score: {p['similarity_score']:.4f}")
        abstract = p.get("abstract", "")
        preview = (abstract[:400] + "…") if len(abstract) > 400 else abstract
        print(f"   {preview}\n")

"""
Patent keyword search + embedding-based reranking.

1) Fetch patents via SerpAPI Google Patents engine.
2) Embed user_description and each patent abstract (snippet) with
   sentence-transformers/mxbai-embed-large-v1 (1024-d output).
3) Rerank by cosine similarity between description embedding and each
   abstract embedding.
"""

# from __future__ import annotations  (duplicate legacy block; commented out)

import json
import os
from typing import Any

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
EMBED_MODEL_NAME = "mxbai-embed-large-v1"

_MODEL: SentenceTransformer | None = None


def _get_serpapi_key() -> str:
    load_dotenv()
    key = os.getenv("SERPAPI_KEY", "").strip()
    if not key:
        raise RuntimeError("SERPAPI_KEY is not set in the environment or .env file")
    return key


def _get_embedder() -> SentenceTransformer:
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    if SentenceTransformer is None:  # pragma: no cover
        raise RuntimeError(
            "sentence-transformers is required but could not be imported."
        ) from _SENTENCE_TRANSFORMERS_IMPORT_ERROR

    _MODEL = SentenceTransformer(EMBED_MODEL_NAME)
    return _MODEL


def _serpapi_search_patents(keyword: str, *, limit: int, timeout: float) -> list[dict[str, str]]:
    api_key = _get_serpapi_key()
    params = {
        "engine": SERPAPI_ENGINE,
        "q": keyword,
        "api_key": api_key,
    }

    try:
        resp = requests.get(SERPAPI_URL, params=params, timeout=timeout)
    except requests.RequestException as e:
        raise RuntimeError(f"SerpAPI request failed: {e}") from e

    if not resp.ok:
        snippet = (resp.text or "")[:800].strip()
        raise RuntimeError(f"SerpAPI HTTP {resp.status_code}: {resp.reason}. {snippet}")

    try:
        data: Any = resp.json()
    except json.JSONDecodeError as e:
        snippet = (resp.text or "")[:200]
        raise RuntimeError(f"SerpAPI returned invalid JSON: {e}. Body starts with: {snippet!r}") from e

    organic = data.get("organic_results") or []
    if not isinstance(organic, list):
        raise RuntimeError("Unexpected SerpAPI response: 'organic_results' is not a list")

    results: list[dict[str, str]] = []
    for item in organic:
        if not isinstance(item, dict):
            continue
        results.append(
            {
                "patent_number": str(item.get("publication_number") or "").strip(),
                "title": str(item.get("title") or "").strip(),
                # SerpAPI provides snippet text; we treat it as "abstract" here.
                "abstract": str(item.get("snippet") or "").strip(),
            }
        )
        if len(results) >= limit:
            break

    return results


def _cosine_similarities(
    description_vec: Any, abstract_vecs: Any
) -> list[float]:
    """
    Assumes vectors are already normalized (cosine similarity == dot product).
    """
    sims: list[float] = []
    for i in range(len(abstract_vecs)):
        sims.append(float((description_vec * abstract_vecs[i]).sum()))
    return sims


def search_patents(
    keyword: str,
    user_description: str,
    *,
    limit: int = 10,
    timeout: float = 30.0,
    max_patents_for_rerank: int = 30,
) -> list[dict[str, Any]]:
    """
    Search Google Patents via SerpAPI, then rerank using embedding cosine similarity.

    Returns top `limit` dicts with:
      - patent_number (from publication_number)
      - title
      - abstract (from SerpAPI snippet)
      - similarity_score (cosine similarity)
    """
    keyword = keyword.strip()
    if not keyword:
        raise ValueError("keyword must be non-empty")
    user_description = (user_description or "").strip()
    if not user_description:
        raise ValueError("user_description must be non-empty")
    if limit < 1:
        raise ValueError("limit must be >= 1")

    # Fetch more than we need, so reranking has headroom.
    initial_limit = max(limit, min(max_patents_for_rerank, 100))
    candidates = _serpapi_search_patents(keyword, limit=initial_limit, timeout=timeout)
    if not candidates:
        return []

    embedder = _get_embedder()

    descriptions = [user_description]
    abstracts = [c.get("abstract", "") or c.get("title", "") or "" for c in candidates]
    texts = descriptions + abstracts

    vecs = embedder.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,  # cosine == dot product
    )
    if vecs.ndim != 2 or vecs.shape[1] != 1024:
        # Keep it strict per your requirement; if the model changes, fail loudly.
        raise RuntimeError(f"Expected 1024-d embeddings, got shape {vecs.shape}")

    description_vec = vecs[0]
    abstract_vecs = vecs[1:]

    scores = _cosine_similarities(description_vec, abstract_vecs)
    for cand, score in zip(candidates, scores):
        cand["similarity_score"] = score

    candidates.sort(key=lambda x: float(x.get("similarity_score", 0.0)), reverse=True)
    return candidates[:limit]


if __name__ == "__main__":
    keyword = "solar energy battery"
    description = (
        "I want to make a solar powered battery storage system for homes"
    )
    results = search_patents(keyword, description, limit=10)
    print(f"Top {len(results)} reranked patents for {keyword!r}:\n")
    for i, p in enumerate(results, 1):
        print(f"{i}. {p['patent_number']} — {p['title']}")
        print(f"   similarity_score: {p['similarity_score']:.4f}")
        abstract = p.get("abstract", "")
        preview = (abstract[:400] + "…") if len(abstract) > 400 else abstract
        print(f"   {preview}\n")

"""
Keyword search for patents via SerpAPI's Google Patents engine.
"""

# from __future__ import annotations  (duplicate legacy block; commented out)

import json
import os
from typing import Any

import requests
from dotenv import load_dotenv

SERPAPI_URL = "https://serpapi.com/search"


def _get_serpapi_key() -> str:
    load_dotenv()
    key = os.getenv("SERPAPI_KEY", "").strip()
    if not key:
        raise RuntimeError("SERPAPI_KEY is not set in the environment or .env file")
    return key


def search_patents(
    keyword: str,
    *,
    limit: int = 10,
    timeout: float = 30.0,
) -> list[dict[str, str]]:
    """
    Search Google Patents via SerpAPI and return up to ``limit`` results.

    Each dict has keys: ``patent_number`` (publication_number), ``title``, and
    ``abstract`` (snippet text).\n    """
    keyword = keyword.strip()
    if not keyword:
        raise ValueError("keyword must be non-empty")
    if limit < 1:
        raise ValueError("limit must be >= 1")

    api_key = _get_serpapi_key()

    params = {
        "engine": "google_patents",
        "q": keyword,
        "api_key": api_key,
    }

    try:
        resp = requests.get(SERPAPI_URL, params=params, timeout=timeout)
    except requests.RequestException as e:
        raise RuntimeError(f"SerpAPI request failed: {e}") from e

    if not resp.ok:
        snippet = (resp.text or "")[:800].strip()
        raise RuntimeError(
            f"SerpAPI HTTP {resp.status_code}: {resp.reason}. {snippet}"
        )

    try:
        data: Any = resp.json()
    except json.JSONDecodeError as e:
        snippet = (resp.text or "")[:200]
        raise RuntimeError(
            f"SerpAPI returned invalid JSON: {e}. Body starts with: {snippet!r}"
        ) from e

    organic = data.get("organic_results") or []
    if not isinstance(organic, list):
        raise RuntimeError(
            f"Unexpected SerpAPI response shape, 'organic_results' is not a list: {data!r}"
        )

    results: list[dict[str, str]] = []
    for item in organic:
        if not isinstance(item, dict):
            continue
        number = str(item.get("publication_number") or "").strip()
        title = str(item.get("title") or "").strip()
        abstract = str(item.get("snippet") or "").strip()
        results.append(
            {
                "patent_number": number,
                "title": title,
                "abstract": abstract,
            }
        )
        if len(results) >= limit:
            break

    return results


if __name__ == "__main__":
    query = "solar energy battery"
    try:
        results = search_patents(query, limit=10)
    except Exception as exc:  # pragma: no cover - simple demo/test
        print(f"Search failed: {exc}")
    else:
        print(f"Top {len(results)} patents for {query!r}:\n")
        for i, p in enumerate(results, 1):
            print(f"{i}. {p['patent_number']} — {p['title']}")
            abstract = p["abstract"]
            preview = (abstract[:400] + "…") if len(abstract) > 400 else abstract
            print(f"   {preview}\n")


# Re-export the embedding/reranking version (later legacy duplicated blocks
# may redefine `search_patents`, so we explicitly wrap here).
def search_patents(
    keyword: str,
    user_description: str,
    *,
    limit: int = 10,
    timeout: float = 30.0,
    max_patents_for_rerank: int = 30,
) -> list[dict[str, Any]]:
    return _SEARCH_PATENTS_EMBEDDING(
        keyword,
        user_description,
        limit=limit,
        timeout=timeout,
        max_patents_for_rerank=max_patents_for_rerank,
    )
