#streamlit run app_streamlit.py
import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# Import your core logic functions
from core.transcriber import transcribe_audio
from core.summarizer import generate_meeting_notes

# Load environment variables (API keys)
load_dotenv()

# --- Utility Functions ---

def display_notes(notes):
    """Displays the structured notes in Streamlit."""
    st.subheader("📝 Meeting Notes")

    # Summary: Handles the "No summary available" case
    st.markdown("### Summary")
    summary_text = notes.get("summary", "No summary available.")
    # Check if summary is a dictionary (which would be an error)
    if isinstance(summary_text, dict):
        st.error("Summary field returned an invalid format.")
    else:
        st.info(summary_text)

    # Action Items
    st.markdown("### Action Items")
    action_items = notes.get("action_items", [])
    if action_items:
        clean_items = []
        for item in action_items:
            # FIX: If the item is still a dictionary (from a model error), format it into a string
            if isinstance(item, dict) and 'owner' in item and 'task' in item:
                clean_items.append(f"**{item['owner']}**: {item['task']}")
            else:
                # Assume the item is already a clean string (the desired format)
                clean_items.append(item)
        
        st.markdown("\n".join([f"- {item}" for item in clean_items]))
    else:
        st.warning("No action items found.")

    # Key Decisions remains correct
    st.markdown("### Key Decisions")
    key_decisions = notes.get("key_decisions", [])
    if key_decisions:
        st.markdown("\n".join([f"- {item}" for item in key_decisions]))
    else:
        st.warning("No key decisions found.")

# --- Main Application UI ---

st.title("🎤 Meeting Notes Extractor")
st.markdown("Powered by **Gemini** (Notes) and **Whisper** (Transcription)")

# ----------------------------------------------------
# 1. File Upload Section
# ----------------------------------------------------

st.header("Upload an Audio File")
uploaded_file = st.file_uploader(
    "Choose a meeting audio file (MP3, WAV, M4A, etc.)", 
    type=["mp3", "wav", "m4a"], 
    accept_multiple_files=False
)


# --- Processing Logic ---

if uploaded_file is not None:
    # Use a Streamlit button to trigger the processing
    if st.button("Process Audio File", key="file_process"):
        
        # Display spinner while processing
        with st.spinner('Processing audio... this may take a few moments.'):
            # 1. Save file to a temporary location for Whisper
            with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
                tmp_file.write(uploaded_file.read())
                file_path = tmp_file.name
            
            try:
                # 2. Transcribe using local Whisper model
                transcript = transcribe_audio(file_path)
                
                if not transcript:
                    st.error("Transcription failed. Please check the audio file and ensure FFmpeg is installed.")
                else:
                    
                    # -----------------------------------------------------------------
                    # 🚀 DISPLAY RAW TRANSCRIPT
                    # -----------------------------------------------------------------
                    st.markdown("---")
                    st.markdown("### 🗣️ Raw Meeting Transcript")
                    st.code(transcript, language='text') 
                    st.markdown("---")
                    # -----------------------------------------------------------------
                    
                    # 3. Generate Notes using Gemini API
                    notes = generate_meeting_notes(transcript)
                    if notes:
                        display_notes(notes)
                    else:
                        st.error("Note generation failed. Check Gemini API key or model response.")
            
            finally:
                # 4. Clean up the temporary file
                os.unlink(file_path)
