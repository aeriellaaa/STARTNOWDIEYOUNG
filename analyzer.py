import os
import json
from dotenv import load_dotenv
import google.generativeai as genai
load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.5-flash")


def extract_ipc_and_keywords(user_description: str) -> dict:
    prompt = f"""
You are a patent classification expert.
Given this invention description, return a JSON object with:
- "ipc_codes": list of 3-5 relevant IPC codes (e.g. "H04L 9/00")
- "search_keywords": list of 8-10 technical keywords for patent search

Respond with ONLY valid JSON, no markdown, no explanation.

Invention: {user_description}
"""
    response = model.generate_content(prompt)
    return json.loads(response.text.strip())


def generate_novelty_report(user_description: str, patent_results: list) -> dict:
    patents_text = "\n\n".join([
        f"Patent {i+1}: {p['title']}\n{p['abstract']}"
        for i, p in enumerate(patent_results[:5])
    ])

    prompt = f"""
You are a patent novelty analyst. Compare this invention against the prior art below.

INVENTION:
{user_description}

PRIOR ART (retrieved patents):
{patents_text}

Return a JSON object with:
- "novelty_score": integer 0-100 (100 = completely novel, 0 = already exists)
- "summary": 2-3 sentence plain English explanation for a non-lawyer
- "closest_patent": title of the most similar patent
- "differentiators": list of 2-4 ways the invention differs from prior art
- "risk_level": one of "Low", "Medium", "High"
- "claim_overlap": list of 2-4 specific claims from prior art that overlap with this invention
- "ipc_class": the single most relevant IPC class for this invention (e.g. "B65D")

Respond with ONLY valid JSON, no markdown, no explanation.
"""
    response = model.generate_content(prompt)
    return json.loads(response.text.strip())

if __name__ == "__main__":
    result = extract_ipc_and_keywords(
        "A foldable water bottle made of biodegradable silicone with a UV sterilisation cap"
    )
    print(result)
def analyze_patents(user_description, patents):
    try:
        report = generate_novelty_report(user_description, patents)
        return report
    except Exception as e:
        return {"error": str(e)}