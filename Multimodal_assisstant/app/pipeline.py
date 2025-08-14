# app/pipeline.py
import io
import google.generativeai as genai
from PIL import Image
from utils import GEMINI_API_KEY

genai.configure(api_key=GEMINI_API_KEY)
# Replace with actual key
model = genai.GenerativeModel("gemini-2.0-flash")

def start_image_chat_session(image_path: str):
    try:
        image = Image.open(image_path)
        chat_session = model.start_chat(history=[])
        return chat_session, image
    except Exception as e:
        return None, f"Error: {e}"

def ask_followup(chat_session, prompt: str, image=None):
    try:
        if image:
            response = chat_session.send_message([prompt, image])
        else:
            response = chat_session.send_message(prompt)
        return response.text
    except Exception as e:
        return f"Error: {e}"
    
def process_text_and_image(text: str, image_bytes: bytes = None) -> str:
    try:
        if image_bytes:
            image = Image.open(io.BytesIO(image_bytes))
            response = model.generate_content([text, image])
        else:
            response = model.generate_content([text])

        return response.text if hasattr(response, "text") else str(response)

    except Exception as e:
        return f"❌ Error: {e}"
    