import gradio as gr
import os
import sys

sys.path.append(os.path.dirname(__file__))

from pdf_reader import extract_text_from_pdf
# Import multi_stage_summarize now
from summarizer import initialize_summarizer, multi_stage_summarize # Removed generate_summary

# --- Global Initialization ---
print("Starting application: Initializing summarizer model...")
initialize_summarizer()

# --- Main Gradio Interface Logic ---

# We need a new function for the Gradio interface that can accept the new parameters
def summarize_document(
    pdf_file,
    min_length_slider: int,
    max_length_slider: int,
    progress=gr.Progress() # Gradio's built-in progress object
) -> str:
    """
    Handles the entire workflow for the Gradio interface, using multi-stage summarization.
    """
    if pdf_file is None:
        return "Please upload a PDF file to summarize."

    pdf_path = pdf_file.name
    print(f"Received PDF file: {pdf_path}")

    progress(0.1, desc="Extracting text from PDF...")
    extracted_text = extract_text_from_pdf(pdf_path)

    if not extracted_text:
        return "Could not extract text from the PDF. Please ensure it's a valid, text-based PDF."
    
    print(f"Extracted {len(extracted_text.split())} words. Starting summarization...")
    
    # Define a simple callback for progress updates
    def summarization_progress_callback(prog_val):
        progress(0.1 + (prog_val * 0.8), desc=f"Summarizing ({int(prog_val*100)}%)...") # Scale to 10-90%

    progress(0.2, desc="Starting AI summarization...")
    final_summary = multi_stage_summarize(
        extracted_text,
        max_model_tokens=1024, # Keep aligned with BART
        overlap_tokens=100,
        summary_min_length=min_length_slider, # Use UI input
        summary_max_length=max_length_slider, # Use UI input
        progress_callback=summarization_progress_callback # Pass the callback
    )
    
    progress(1.0, desc="Summarization complete!")

    if "Error:" in final_summary or "No text provided" in final_summary or "Failed to summarize" in final_summary:
        print(f"Summarization error: {final_summary}")
        return f"An error occurred during summarization: {final_summary}"

    print("Final summarization complete.")
    return final_summary

# --- Gradio Interface Definition ---

# Define the Gradio interface components
pdf_input = gr.File(label="Upload PDF Document", type="filepath", file_types=[".pdf"])
min_length_slider = gr.Slider(minimum=20, maximum=300, value=50, step=10, label="Minimum Summary Length (words)")
max_length_slider = gr.Slider(minimum=50, maximum=500, value=200, step=10, label="Maximum Summary Length (words)")
summary_output = gr.Textbox(label="Generated Summary", lines=15, interactive=False) # Increased lines for longer summaries

# Create the Gradio interface
demo = gr.Interface(
    fn=summarize_document, # Use the new function
    inputs=[pdf_input, min_length_slider, max_length_slider], # Pass all inputs
    outputs=summary_output,
    title="AI-Powered PDF Summarizer",
    description="Upload a PDF document to get a concise summary generated by advanced AI. This version handles larger documents by chunking and provides configurable summary lengths.",
    allow_flagging="never",
    # Examples can be useful here, but need accessible PDF paths
    # examples=[["path/to/your/sample_small.pdf"], ["path/to/your/sample_large.pdf"]],
    # To use examples, uncomment above and provide paths to actual PDF files in your project directory
    # or accessible via URL.
)

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch()