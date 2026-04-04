---
title: AviShkar
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.0.0"
app_file: app.py
pinned: false
---
```markdown
#  AviShkar — Know Before You Build

AI-powered prior art patent search platform. Describe your invention idea in 
plain English and AviShkar searches existing patents and generates a novelty 
report instantly — no legal expertise needed.

---

##  What It Does

1. Describe your invention idea in plain English
2. AviShkar searches Google Patents via SerpAPI
3. Gemini AI analyzes results and generates a novelty report with:
   - Novelty score (0-100)
   - Most similar existing patents
   - Differentiating features
   - Risk level
   - IPC class recommendation

---

##  Tech Stack

- Frontend: HTML + CSS + JS (static)
- Backend: FstAPI + Python
- Patent Search:SerpAPI (Google Patents)
- AI Analysis:Gemini 2.5 Flash (Google AI Studio)
- Embeddings:mxbai-embed-large-v1 (SentenceTransformers)
- Similarity Ranking:Cosine similarity

---

##  Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/aeriellaaa/Avishkar.git
cd Avishkar
```

### 2. Install dependencies
```bash
pip install fastapi uvicorn python-dotenv google-generativeai sentence-transformers requests gradio
```

### 3. Create a `.env` file in the root folder
```
GEMINI_API_KEY=your_gemini_key_here
SERPAPI_KEY=your_serpapi_key_here
```

Get your keys here:
- Gemini: https://aistudio.google.com
- SerpAPI: https://serpapi.com

### 4. Run the backend (Terminal 1)
```bash
uvicorn api_server:app --reload --host 127.0.0.1 --port 8000
```

### 5. Serve the frontend (Terminal 2)
```bash
python -m http.server 8080
```

### 6. Open in browser
```
http://localhost:8080
```

---

## 📁 Project Structure

```
AviShkar/
├── index.html          # Main frontend
├── styles.css          # Styling
├── script.js           # Frontend logic
├── api_server.py       # FastAPI backend
├── search.py           # Patent search via SerpAPI
├── analyzer.py         # Gemini AI analysis
├── ui/
│   └── theme.py        # Gradio theme (optional)
├── requirements.txt    # Dependencies
├── .env                # API keys (never commit this)
└── .gitignore
```

---

## 👥 Team — Hope it Compiles

| Name | Role |
|------|------|
| Srishti Srivastava | Team Lead · Patent Search Module · API Server |
| Malavika Rajeev Nair | Frontend · UI Design |
| Akshata Srivastava | App Integration · Glue Code |
| Palak Lohia | Gemini AI Analysis Module |

---

## 🏆 Built For

OssomeHacks — GitHub Community SRM
Track: Open Innovation using AI/ML

---

