

```markdown
# AviShkar — Know Before You Build

AI-powered prior art patent search platform. Describe your invention idea in plain English and AviShkar will search existing patents and generate a novelty report instantly — no legal expertise needed.

---

## What It Does

1. You describe your invention idea in plain English
2. AviShkar searches Google Patents via SerpAPI
3. Gemini AI analyzes results and generates a novelty report with:
   - Novelty score (0-100)
   - Most similar existing patents
   - Differentiating features
   - Risk level
   - IPC class recommendation

---

##  Tech Stack

- Frontend: Gradio
- Backend:Python
- Patent Search: SerpAPI (Google Patents)
- AI Analysis: Gemini 2.5 Flash (Google AI Studio)
- Embeddings: mxbai-embed-large-v1 (SentenceTransformers)
- Similarity Ranking: FAISS-style cosine similarity

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/aeriellaaa/Avishkar.git
cd Avishkar
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Create `.env` file
```
GEMINI_API_KEY=your_gemini_key_here
SERPAPI_KEY=your_serpapi_key_here
```

### 4. Run
```bash
python app.py
```

Open http://127.0.0.1:7860 in your browser.

---

##  Team — Hope it Compiles

| Name | Role |
|------|------|
| Srishti Srivastava | Team Lead, Patent Search Module |
| Malavika Rajeev Nair | Frontend & UI Theme |
| Akshata Srivastava | App Integration & Glue Code |
| Palak Lohia | Gemini AI Analysis Module |

---

##  Built For

OssomeHacks — GitHub Community SRM
Track: Open Innovation using AI/ML

---


