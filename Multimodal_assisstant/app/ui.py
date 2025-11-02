import streamlit as st
import os
from dotenv import load_dotenv
from pipeline import process_text_and_image
from chat import ChatSession

# ----------------------------
# Load environment
# ----------------------------
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# ----------------------------
# Configure Streamlit
# ----------------------------
st.set_page_config(page_title="Multimodal Assistant", layout="wide")

st.title("🧠 Multimodal Assistant (Gemini 2.0 Flash)")

# Tabs for Text and Image assistants
tab1, tab2 = st.tabs(["💬 Text Assistant", "🖼️ Image Assistant"])

# Initialize chat sessions
if "text_session" not in st.session_state:
    st.session_state.text_session = ChatSession()

if "image_session" not in st.session_state:
    st.session_state.image_session = ChatSession()

# Initialize chat histories
if "text_messages" not in st.session_state:
    st.session_state.text_messages = []

if "image_messages" not in st.session_state:
    st.session_state.image_messages = []

# ----------------------------
# 💬 TEXT ASSISTANT TAB
# ----------------------------
with tab1:
    st.subheader("Text Assistant")

    # Display chat history
    for msg in st.session_state.text_messages:
        st.markdown(f"**{msg['role'].capitalize()}:** {msg['content']}")

    # User input
    text_input = st.chat_input("Ask a text-only question here")

    if text_input:
        # Append user message
        st.session_state.text_messages.append({"role": "user", "content": text_input})

        # Generate response
        response = process_text_and_image(text_input)  # Text-only mode
        st.session_state.text_messages.append({"role": "assistant", "content": response})

        # Refresh UI
        st.rerun()

# ----------------------------
# 🖼️ IMAGE ASSISTANT TAB
# ----------------------------
with tab2:
    st.subheader("Image Assistant")

    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        # Read image bytes and store once
        st.session_state.image_bytes = uploaded_file.read()
        st.session_state.image_session.handle_upload(uploaded_file)
        st.image(uploaded_file, caption="Uploaded Image", width='stretch')

    # Display chat history
    for msg in st.session_state.image_messages:
        st.markdown(f"**{msg['role'].capitalize()}:** {msg['content']}")

    # User input
    image_input = st.chat_input("Ask a question about your image")

    if image_input:
        st.session_state.image_messages.append({"role": "user", "content": image_input})

        # Ensure image bytes exist before analysis
        image_bytes = st.session_state.get("image_bytes", None)
        response = process_text_and_image(image_input, image_bytes)

        # Append and display response
        st.session_state.image_messages.append({"role": "assistant", "content": response})
        st.rerun()
