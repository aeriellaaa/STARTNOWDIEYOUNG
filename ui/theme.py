import gradio as gr

import gradio as gr

def get_theme():
    return gr.themes.Soft(
        primary_hue=gr.themes.Color(
            c50="#f0f9ff", c100="#e0f2fe", c200="#bae6fd",
            c300="#7dd3fc", c400="#38bdf8", c500="#0ea5e9",
            c600="#0284c7", c700="#0369a1", c800="#075985",
            c900="#0c4a6e", c950="#082f49",
        ),
        neutral_hue=gr.themes.Color(
            c50="#f8fafc", c100="#f1f5f9", c200="#e2e8f0",
            c300="#cbd5e1", c400="#94a3b8", c500="#64748b",
            c600="#475569", c700="#334155", c800="#1e293b",
            c900="#0f172a", c950="#020617",
        ),
        font=[gr.themes.GoogleFont("Outfit"), "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
    ).set(
        body_background_fill="#0f172a",
        body_text_color="#f8fafc",
        block_background_fill="#1e293b",
        block_border_color="#334155",
        block_border_width="1px",
        block_label_text_color="#94a3b8",
        block_label_background_fill="#0f172a",
        input_background_fill="#0f172a",
        input_border_color="#334155",
        button_primary_background_fill="#0ea5e9",
        button_primary_text_color="#ffffff",
        button_primary_background_fill_hover="#0284c7",
        button_secondary_background_fill="transparent",
        button_secondary_border_color="#334155",
        button_secondary_text_color="#f8fafc",
    )