"""
Keyword search for patents via SerpAPI's Google Patents engine.
"""

from __future__ import annotations

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
