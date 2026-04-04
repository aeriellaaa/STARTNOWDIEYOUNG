"""
REST API for the static frontend (index.html + script.js).

Run (from repo root, with venv + .env for SERPAPI_KEY and GEMINI_API_KEY):
  uvicorn api_server:app --reload --host 127.0.0.1 --port 8000

Then set in script.js: const API_BASE_URL = 'http://localhost:8000';
Serve the HTML/CSS/JS folder separately (e.g. py -m http.server 8080).
"""
from __future__ import annotations

import re
from typing import Any, Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from analyzer import analyze_patents
from search import search_patents
from fastapi.staticfiles import StaticFiles

app.mount("/assets", StaticFiles(directory="assets"), name="assets")

app = FastAPI(title="AviShkar API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _keyword_from_text(text: str) -> str:
    return " ".join(text.strip().split()[:5])


def _year_from_patent_id(pid: str) -> str:
    m = re.search(r"(19|20)\d{2}", pid or "")
    return m.group(0) if m else "—"


def _map_patent(p: dict[str, Any], report: dict[str, Any], idx: int) -> dict[str, Any]:
    sim = float(p.get("similarity_score", 0))
    novelty_pct = report.get("novelty_score") or max(0, min(100, int(round(sim * 100))))
    diffs = report.get("differentiators") or []
    diff_text = " • ".join(diffs) if diffs else (report.get("summary") or "Compare your idea with this prior art.")
    return {
        "id": p.get("patent_number") or f"Result-{idx + 1}",
        "title": p.get("title") or "Untitled",
        "db": "Google Patents",
        "ipc": str(report.get("ipc_class") or "—"),
        "year": _year_from_patent_id(str(p.get("patent_number", ""))),
        "novelty": novelty_pct,
        "abstract": p.get("abstract") or "",
        "diff": diff_text,
    }


@app.post("/search")
async def search(
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
):
    # Sketch-only not supported by current pipeline; description drives SerpAPI + embeddings
    _ = file  # reserved for future vision step

    body = (text or "").strip()
    if not body:
        return {"results": [], "error": "Add a text description of your idea (sketch-only search not wired yet)."}

    keyword = _keyword_from_text(body)
    try:
        patents = search_patents(keyword, body)
    except Exception as e:
        return {"results": [], "error": str(e)}

    if not patents:
        return {"results": [], "error": "No patents returned from search."}

    report = analyze_patents(body, patents)
    if report.get("error"):
        report = {
            "differentiators": [],
            "summary": str(report.get("error")),
            "ipc_class": "—",
        }

    results = [_map_patent(p, report, i) for i, p in enumerate(patents)]
    return {"results": results}


@app.get("/health")
def health():
    return {"ok": True}