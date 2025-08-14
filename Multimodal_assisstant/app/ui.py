import streamlit as st
import os
from pipeline import process_text_and_image
from dotenv import load_dotenv

# Load environment variables from .env file in the parent directory
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
st.set_page_config(page_title="Multimodal Assistant", layout="wide")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🧠 Multimodal Assistant")

# Upload image (optional for multimodal input)
uploaded_file = st.file_uploader(
    "Upload an image", type=["png", "jpg", "jpeg"], label_visibility="collapsed"
)

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Assistant:** {msg['content']}")

# Input box at the bottom
user_input = st.chat_input("Ask a question")

if user_input:
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Read image bytes if uploaded
    image_bytes = uploaded_file.read() if uploaded_file else None

    # Get assistant response
    assistant_response = process_text_and_image(user_input, image_bytes)

    # Add assistant response to session state
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})

    # Force rerun to immediately show new messages
    st.rerun()
