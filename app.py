import gradio as gr
from ui.theme import get_theme
from search import search_patents
from analyzer import analyze_patents

# Custom CSS for modern dashboard feel
custom_css = """
.dashboard-container {
    padding: 20px;
}
.sidebar-panel {
    background: #1e293b;
    border-radius: 12px;
    padding: 20px;
    border: 1px solid #334155;
}
.patent-card {
    background: #1e293b;
    border-radius: 10px;
    padding: 15px;
    margin-bottom: 12px;
    border: 1px solid #334155;
    transition: all 0.2s ease;
}
.patent-card:hover {
    border-color: #0ea5e9;
    transform: translateY(-2px);
}
.score-gauge {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 20px;
}
.risk-badge {
    padding: 4px 12px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.8em;
}
.risk-low { background: #065f46; color: #34d399; }
.risk-medium { background: #92400e; color: #fbbf24; }
.risk-high { background: #991b1b; color: #f87171; }
"""

def create_gauge_html(score):
    color = "#0ea5e9"
    if score < 30: color = "#f87171"
    elif score < 70: color = "#fbbf24"
    else: color = "#34d399"
    
    return f"""
    <div class="score-gauge">
        <svg width="120" height="120" viewBox="0 0 120 120">
            <circle cx="60" cy="60" r="54" fill="none" stroke="#334155" stroke-width="8" />
            <circle cx="60" cy="60" r="54" fill="none" stroke="{color}" stroke-width="8" 
                stroke-dasharray="{3.39 * score} 339" stroke-linecap="round" transform="rotate(-90 60 60)" />
            <text x="60" y="65" text-anchor="middle" font-size="24" font-weight="bold" fill="white">{score}%</text>
        </svg>
        <div style="margin-top: 10px; font-weight: 600; color: #94a3b8;">NOVELTY SCORE</div>
    </div>
    """

def format_patents_html(patents):
    if not patents:
        return "<div style='text-align: center; color: #94a3b8; padding: 40px;'>No results to display yet.</div>"
    
    html = "<div style='display: flex; flex-direction: column; gap: 12px;'>"
    for p in patents:
        score_pct = int(p.get('similarity_score', 0) * 100)
        html += f"""
        <div class="patent-card">
            <div style="display: flex; justify-content: space-between; align-items: start;">
                <h4 style="margin: 0; color: #f8fafc;">{p['title']}</h4>
                <span style="color: #0ea5e9; font-weight: bold;">{score_pct}% Match</span>
            </div>
            <p style="color: #cbd5e1;">{p['abstract']}</p>
        </div>
        """
    html += "</div>"
    return html

def run_avishkar(user_description):
    if not user_description.strip():
        return None, "Please enter a description.", "", "", "", ""
    
    keyword = " ".join(user_description.strip().split()[:5])
    patents = search_patents(keyword, user_description)
    report = analyze_patents(user_description, patents)
    
    return create_gauge_html(report.get('novelty_score', 0)), format_patents_html(patents), "", "", ""

# UI
with gr.Blocks(theme=get_theme(), css=custom_css, title="AviShkar — Patent Search AI") as demo:
    with gr.Row(elem_classes=["dashboard-container"]):
        
        with gr.Column(scale=1):
            
            # 🔥 UPDATED MARKDOWN (ONLY CHANGE)
            gr.Markdown(
                """
                <div style="display:flex; align-items:center; gap:12px;">
                    <img src="https://raw.githubusercontent.com/aeriellaaa/Avishkar/main/logo_bg_removed.png.png" width="60">
                    <div>
                        <h1 style="margin:0;">🚀 AviShkar</h1>
                        <p style="margin:0; color:#94a3b8;">Know Before You Build</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            user_input = gr.Textbox(lines=5, placeholder="Describe your idea...")
            analyze_btn = gr.Button("Analyze")

        with gr.Column(scale=2):
            results = gr.HTML()

    analyze_btn.click(run_avishkar, inputs=user_input, outputs=results)

if __name__ == "__main__":
    demo.launch()
    