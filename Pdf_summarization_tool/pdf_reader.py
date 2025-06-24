import fitz  # PyMuPDF
import os

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts all text from a given PDF file.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: The extracted text, or an empty string if an error occurs.
    """
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at '{pdf_path}'")
        return ""

    if not pdf_path.lower().endswith('.pdf'):
        print(f"Error: Provided file '{os.path.basename(pdf_path)}' is not a PDF.")
        return ""

    text = ""
    try:
        # Open the PDF document
        document = fitz.open(pdf_path)

        # Iterate through each page
        for page_num in range(document.page_count):
            page = document.load_page(page_num)
            text += page.get_text() + "\n" # Add a newline between pages for readability

        document.close() # Close the document after processing
    except Exception as e:
        print(f"An error occurred while extracting text from '{pdf_path}': {e}")
        return ""
    
    # Simple cleaning: remove multiple consecutive newlines, though models handle some noise
    text = os.linesep.join([s for s in text.splitlines() if s.strip()])

    return text
