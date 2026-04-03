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

/* Custom Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-track {
    background: #0f172a;
}
::-webkit-scrollbar-thumb {
    background: #334155;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #475569;
}
"""

def create_gauge_html(score):
    # Circular SVG Gauge
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
                <h4 style="margin: 0; color: #f8fafc; font-size: 1.1em;">{p['title']}</h4>
                <span style="color: #0ea5e9; font-weight: bold;">{score_pct}% Match</span>
            </div>
            <div style="color: #94a3b8; font-size: 0.85em; margin: 4px 0;">{p['patent_number']}</div>
            <p style="color: #cbd5e1; font-size: 0.9em; margin: 8px 0; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;">
                {p['abstract']}
            </p>
        </div>
        """
    html += "</div>"
    return html

def run_avishkar(user_description):
    if not user_description.strip():
        return None, "Please enter a description.", "", "", "", ""
    
    # Keyword extraction (simplified)
    keyword = " ".join(user_description.strip().split()[:5])
    
    # Step 1: Search
    patents = search_patents(keyword, user_description)
    if not patents:
        return None, "No patents found.", "", "", "", ""
    
    # Step 2: Analyze
    report = analyze_patents(user_description, patents)
    
    # Format Results
    results_html = format_patents_html(patents)
    gauge_html = create_gauge_html(report.get('novelty_score', 0))
    
    risk_class = f"risk-{report.get('risk_level', 'Medium').lower()}"
    analysis_sidebar = f"""
    <div style="padding: 10px;">
        <h3 style="margin-top: 0;">AI Analysis</h3>
        <p style="color: #cbd5e1; line-height: 1.5;">{report.get('summary', '')}</p>
        
        <div style="margin: 20px 0;">
            <div style="color: #94a3b8; font-size: 0.8em; margin-bottom: 5px;">RISK LEVEL</div>
            <span class="risk-badge {risk_class}">{report.get('risk_level', 'N/A')}</span>
        </div>
        
        <div style="margin: 20px 0;">
            <div style="color: #94a3b8; font-size: 0.8em; margin-bottom: 5px;">CLOSEST PRIOR ART</div>
            <div style="color: #f8fafc; font-weight: 500;">{report.get('closest_patent', 'N/A')}</div>
        </div>
        
        <div style="margin: 20px 0;">
            <div style="color: #94a3b8; font-size: 0.8em; margin-bottom: 5px;">IPC CLASS</div>
            <span style="background: #334155; color: #f8fafc; padding: 2px 8px; border-radius: 4px; font-family: monospace;">{report.get('ipc_class', 'N/A')}</span>
        </div>
    </div>
    """
    
    differentiators_md = "### ✅ Key Differentiators\n" + "\n".join([f"- {d}" for d in report.get('differentiators', [])])
    
    return gauge_html, results_html, analysis_sidebar, differentiators_md

with gr.Blocks(theme=get_theme(), css=custom_css, title="AviShkar — Patent Search AI") as demo:
    with gr.Row(elem_classes=["dashboard-container"]):
        # LEFT COLUMN - Input & Info
        with gr.Column(scale=1):
            gr.Markdown("# 🚀 AviShkar\n**Know Before You Build**")
            user_input = gr.Textbox(
                lines=8,
                placeholder="Describe your invention idea here in plain English... e.g., 'A solar powered backpack with integrated cooling fans and wireless charging...'",
                label="Your Idea",
                elem_id="idea-input"
            )
            analyze_btn = gr.Button("Analyze Invention", variant="primary")
            
            gr.Markdown("""
            ### 💡 Tips
            - Be specific about technical novelties.
            - Mention materials and mechanisms.
            - Describe the problem it solves.
            """)
            
        # MIDDLE COLUMN - Search Results
        with gr.Column(scale=2):
            gr.Markdown("### 🔍 Prior Art Results")
            results_feed = gr.HTML(format_patents_html([]))
            
        # RIGHT COLUMN - AI Analysis
        with gr.Column(scale=1, elem_classes=["sidebar-panel"]):
            score_display = gr.HTML(create_gauge_html(0))
            analysis_panel = gr.HTML("<div style='color: #94a3b8; padding: 20px;'>Run analysis to see AI insights.</div>")
            diff_markdown = gr.Markdown("")

    analyze_btn.click(
        fn=run_avishkar,
        inputs=[user_input],
        outputs=[score_display, results_feed, analysis_panel, diff_markdown]
    )

if __name__ == "__main__":
    demo.launch()
