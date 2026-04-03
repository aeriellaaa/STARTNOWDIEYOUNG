import gradio as gr
<<<<<<< HEAD
from search import search_patents
from analyzer import analyze_patents

def run_avishkar(user_description):
    if not user_description.strip():
        return "Please enter a description of your idea."
    
    # Use first 5 words as keyword
    keyword = " ".join(user_description.strip().split()[:5])
    
    # Step 1: Search patents
    patents = search_patents(keyword, user_description)
    
    if not patents:
        return "No patents found. Try a different description."
    
    # Step 2: Analyze with Gemini
    result = analyze_patents(user_description, patents)
    
    return result

demo = gr.Interface(
    fn=run_avishkar,
    inputs=gr.Textbox(
        lines=5,
        placeholder="Describe your invention idea here in plain English...",
        label="Your Idea"
    ),
    outputs=gr.Markdown(label="AviShkar Analysis"),
    title="🚀 AviShkar — Know Before You Build",
    description="AI-powered prior art search. Describe your idea and we'll check if it already exists."
)

if __name__ == "__main__":
    demo.launch()
=======
import time

# ── BACKEND STUB ──────────────────────────────

def run_pipeline(sketch_image, idea_text, domain):
    time.sleep(2)

    results_md = """
### 🥇 Match 1 — Similarity: 92%
**Multimodal patent retrieval using joint sketch-text embedding**

`IPC: G06F 16/903` · USPTO · US11,237,845 · 2021

A system that fuses visual and textual modalities to retrieve semantically similar patents without legal expertise.

[View on Google Patents](https://patents.google.com)

---

### 🥈 Match 2 — Similarity: 78%
**Automated feature extraction from technical drawings**

`IPC: G06V 10/764` · EPO · EP3821401 · 2020

Convolutional vision transformers for extracting structural features from engineering sketches.

[View on Google Patents](https://patents.google.com)

---

### 🥉 Match 3 — Similarity: 64%
**Natural language interface for patent retrieval systems**

`IPC: G06F 40/30` · WIPO · WO2019215432 · 2019

NLP-based query expansion that maps plain descriptions to IPC codes.

[View on Google Patents](https://patents.google.com)
"""

    novelty_md = """
## Novelty Score: 74 / 100

**Assessment:** Moderate novelty detected.

Your invention shows meaningful differentiation in the democratisation angle and the combined sketch and text pipeline, but the core multimodal fusion approach has prior coverage.

**Strengths:**
- Real-time IPC auto-classification is differentiating
- Combined visual-textual query vector is novel in this domain
- MSME-focused access model has no direct prior art

**Claim areas to strengthen:**
- Real-time sketch feature extraction speed
- The specific weighted concat + L2 norm fusion method
"""
    return results_md, novelty_md


# ── PIPELINE ─────────────────────────────────

PIPELINE_STEPS = [
    ("Vision feature extraction", "OpenCLIP ViT-H-14"),
    ("IPC code mapping",          "NLP classifier"),
    ("Semantic embedding",        "mxbai-embed-large-v1"),
    ("Vector retrieval",          "FAISS + Qdrant"),
    ("Reranking",                 "ms-marco-MiniLM"),
    ("Novelty analysis",          "Gemini 2.5 Flash"),
]

def make_pipeline_md(active_step=-1, done=False):
    lines = []
    for i, (name, model) in enumerate(PIPELINE_STEPS):
        if done or i < active_step:
            prefix = "✅"
        elif i == active_step:
            prefix = "⏳"
        else:
            prefix = "⬜"
        lines.append(f"{prefix} **{name}** — `{model}`")
    return "\n\n".join(lines)


def search_with_progress(sketch, idea, domain):
    if not idea.strip():
        yield make_pipeline_md(), "Please describe your invention first.", ""
        return
    for step in range(len(PIPELINE_STEPS)):
        yield make_pipeline_md(active_step=step), "_Processing, please wait..._", ""
        time.sleep(0.9)
    results_md, novelty_md = run_pipeline(sketch, idea, domain)
    yield make_pipeline_md(done=True), results_md, novelty_md


# ── CSS ───────────────────────────────────────

CSS = """
* { box-sizing: border-box; }

body, .gradio-container, .main, .wrap {
    background-color: #f5f0e8 !important;
    color: #0f0e0c !important;
}

/* kill dark theme everywhere */
.dark, [data-testid] {
    --background-fill-primary: #f5f0e8 !important;
    --background-fill-secondary: #ede7d6 !important;
    --color-accent: #c8922a !important;
    --body-text-color: #0f0e0c !important;
    --block-label-text-color: #6b6559 !important;
}

/* all text default dark */
p, span, label, div, h1, h2, h3, h4, li, td {
    color: #0f0e0c !important;
}

/* markdown text */
.prose p, .prose li, .prose h1, .prose h2, .prose h3,
.md p, .md li, .md h1, .md h2, .md h3 {
    color: #0f0e0c !important;
}

/* inputs */
textarea, input, select {
    background: #ede7d6 !important;
    color: #0f0e0c !important;
    border: 1.5px solid #c8bfaa !important;
    border-radius: 4px !important;
}

textarea::placeholder, input::placeholder {
    color: #9a9080 !important;
}

/* blocks / panels */
.block, .panel, .form, .gap, .padded {
    background: #f5f0e8 !important;
    border-color: #c8bfaa !important;
}

/* labels above inputs */
.block label span, .svelte-1gfkn6j {
    color: #6b6559 !important;
    font-size: 0.82rem !important;
}

/* button */
button.primary, button[variant="primary"], .btn-primary {
    background: #0f0e0c !important;
    color: #f5f0e8 !important;
    border: none !important;
    border-radius: 4px !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
}
button.primary:hover {
    background: #1a6b5c !important;
}

/* image upload box */
.upload-container, [data-testid="image"], .image-container {
    background: #ede7d6 !important;
    border: 1.5px dashed #c8bfaa !important;
    border-radius: 6px !important;
    color: #0f0e0c !important;
}

/* dropdown */
.wrap-inner, .multiselect, select {
    background: #ede7d6 !important;
    color: #0f0e0c !important;
}

/* code blocks inside markdown */
code {
    background: #0f0e0c !important;
    color: #c8922a !important;
    padding: 2px 6px !important;
    border-radius: 3px !important;
    font-size: 0.82rem !important;
}

/* stat boxes */
.stat-box {
    background: #ede7d6;
    border: 1px solid #c8bfaa;
    border-radius: 6px;
    padding: 1.2rem;
    text-align: center;
    width: 100%;
}

.stat-number {
    font-size: 1.8rem;
    font-weight: 800;
    color: #0f0e0c !important;
    line-height: 1;
}

.stat-gold {
    color: #c8922a !important;
}

.stat-label {
    font-size: 0.65rem;
    color: #6b6559 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 6px;
}

/* header */
#avishkar-header {
    background: #0f0e0c;
    padding: 1.4rem 2rem;
    border-radius: 8px;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

#avishkar-header h1 {
    font-size: 2rem;
    font-weight: 800;
    margin: 0;
    color: #f5f0e8 !important;
    letter-spacing: -0.03em;
}

#avishkar-header h1 span {
    color: #c8922a !important;
}

.header-sub {
    font-size: 0.72rem;
    color: rgba(255,255,255,0.5) !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 0.3rem;
}

.header-right {
    font-size: 0.7rem;
    color: rgba(255,255,255,0.4) !important;
    text-align: right;
    line-height: 1.8;
}

/* footer */
#footer-bar {
    text-align: center;
    font-size: 0.68rem;
    color: #6b6559 !important;
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid #c8bfaa;
    letter-spacing: 0.06em;
}

/* divider */
.divider {
    border: none;
    border-top: 1px solid #c8bfaa;
    margin: 1rem 0;
}

/* row gap fix */
.row-wrap {
    gap: 1rem !important;
}
"""


# ── BUILD UI ──────────────────────────────────

def build_ui():

    with gr.Blocks(
        css=CSS,
        title="AviShkar — Know Before You Build",
        theme=gr.themes.Base(
            primary_hue="orange",
            neutral_hue="stone",
        )
    ) as demo:

        # HEADER
        gr.HTML("""
        <div id="avishkar-header">
            <div>
                <h1>Avi<span>Shkar</span></h1>
                <div class="header-sub">Know before you build &nbsp;·&nbsp; AI-powered prior art search</div>
            </div>
            <div class="header-right">
                OSSOMEHACKS<br/>
                Open Innovation Track<br/>
                Team: Hope it Compiles
            </div>
        </div>
        """)

        # STATS
        with gr.Row():
            gr.HTML("""<div class="stat-box">
                <div class="stat-number">50k<span class="stat-gold">+</span></div>
                <div class="stat-label">Patents Indexed</div>
            </div>""")
            gr.HTML("""<div class="stat-box">
                <div class="stat-number">60<span class="stat-gold">s</span></div>
                <div class="stat-label">Analysis Time</div>
            </div>""")
            gr.HTML("""<div class="stat-box">
                <div class="stat-number">Rs<span class="stat-gold">0</span></div>
                <div class="stat-label">Agent Fees</div>
            </div>""")
            gr.HTML("""<div class="stat-box">
                <div class="stat-number">3<span class="stat-gold">+</span></div>
                <div class="stat-label">Databases</div>
            </div>""")

        gr.HTML("<div class='divider'></div>")

        # MAIN
        with gr.Row(equal_height=False):

            with gr.Column(scale=1):
                gr.Markdown("### Your Invention")

                sketch_input = gr.Image(
                    label="Sketch / Diagram (optional)",
                    type="pil",
                    sources=["upload", "clipboard"],
                    height=220,
                )

                idea_input = gr.Textbox(
                    label="Describe your invention",
                    placeholder="Describe in plain English — what does it do, what problem does it solve, what makes it unique?",
                    lines=6,
                )

                domain_input = gr.Dropdown(
                    label="Domain / Field (optional)",
                    choices=[
                        "Auto-detect",
                        "Mechanical Engineering",
                        "Electrical & Electronics",
                        "Software & Computing",
                        "Biotechnology & Pharma",
                        "Chemical Engineering",
                        "Medical Devices",
                        "Renewable Energy",
                        "Consumer Products",
                        "Agriculture & Food",
                    ],
                    value="Auto-detect",
                )

                search_btn = gr.Button("Search Prior Art", variant="primary", size="lg")

                gr.Markdown("### Pipeline Status")
                pipeline_status = gr.Markdown(value=make_pipeline_md())

            with gr.Column(scale=1):
                gr.Markdown("### Prior Art Matches")
                results_output = gr.Markdown(value="_Results will appear here after search._")

                gr.HTML("<div class='divider'></div>")

                gr.Markdown("### Novelty Report")
                novelty_output = gr.Markdown(value="_Novelty analysis will appear here._")

        search_btn.click(
            fn=search_with_progress,
            inputs=[sketch_input, idea_input, domain_input],
            outputs=[pipeline_status, results_output, novelty_output],
        )

        gr.HTML("""
        <div id="footer-bar">
            TEAM: SRISHTI &nbsp;·&nbsp; MALAVIKA &nbsp;·&nbsp; AKSHATA &nbsp;·&nbsp; PALAK
            &nbsp;&nbsp;|&nbsp;&nbsp;
            HOPE IT COMPILES &nbsp;·&nbsp; OSSOMEHACKS
        </div>
        """)

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
    )
>>>>>>> 1ea3fc16dd1c1f2727a70fab7c76bdc478edbf11
