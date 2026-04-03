import gradio as gr

def get_theme():
    return gr.themes.Base(
        primary_hue=gr.themes.Color(
            c50="#faf5ea", c100="#f5f0e8", c200="#ede7d6",
            c300="#c8bfaa", c400="#c8922a", c500="#c8922a",
            c600="#b87a1a", c700="#9a6512", c800="#7a500e",
            c900="#5a3c0a", c950="#3a2806",
        ),
        neutral_hue=gr.themes.Color(
            c50="#f5f0e8", c100="#ede7d6", c200="#c8bfaa",
            c300="#9a9080", c400="#6b6559", c500="#4a4540",
            c600="#333028", c700="#1e1c18", c800="#141210",
            c900="#0f0e0c", c950="#0a0908",
        ),
        font=[gr.themes.GoogleFont("Syne"), "Georgia", "serif"],
        font_mono=[gr.themes.GoogleFont("DM Mono"), "monospace"],
    ).set(
        body_background_fill="#f5f0e8",
        body_text_color="#0f0e0c",
        block_background_fill="#ede7d6",
        block_border_color="#c8bfaa",
        block_border_width="1.5px",
        block_label_text_color="#6b6559",
        block_label_background_fill="#f5f0e8",
        input_background_fill="#ede7d6",
        input_border_color="#c8bfaa",
        button_primary_background_fill="#0f0e0c",
        button_primary_text_color="#f5f0e8",
        button_primary_background_fill_hover="#1a6b5c",
        button_secondary_background_fill="transparent",
        button_secondary_border_color="#c8bfaa",
        button_secondary_text_color="#0f0e0c",
    )