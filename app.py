import gradio as gr
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
