import os
import io
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("❌ GEMINI_API_KEY not found. Please set it in your .env file.")

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.0-flash")

def process_text_and_image(text: str, image_bytes: bytes = None) -> str:
    """Handles both text and multimodal (text + image) inputs."""
    try:
        if image_bytes:
            image = Image.open(io.BytesIO(image_bytes))
            response = model.generate_content([text, image])
        else:
            response = model.generate_content([text])

        return response.text if hasattr(response, "text") else str(response)

    except Exception as e:
        return f"❌ Error: {e}"
