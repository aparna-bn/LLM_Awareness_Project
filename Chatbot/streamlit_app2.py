# streamlit run streamlit_app2.py
import streamlit as st
import requests
import json

# --- Configuration ---
# URL for your FastAPI backend - CONFIRM THIS IS STILL CORRECT (e.g., http://localhost:8001)
FASTAPI_URL = "http://localhost:8001" # Make sure this matches your backend's running port
CHAT_ENDPOINT = f"{FASTAPI_URL}/chat"

# --- Streamlit UI Setup ---
st.set_page_config(page_title="GeminiSpeak", page_icon="💬", layout="centered")
st.title("💬Real-Time AI Assistant")
st.markdown("Ask anything! Gemini is here to chat.")

# Initialize chat history in Streamlit's session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display Existing Chat Messages with Avatars ---
# Iterate through messages and display them with appropriate avatars
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Clear Chat Button (New Feature) ---
# Add a button to clear the chat history. Placed at the top or sidebar for easy access.
# We use st.session_state.clear() to remove all items, then re-initialize.
if st.sidebar.button("Clear Chat"): # Placed in the sidebar for a cleaner main layout
    st.session_state.messages = []
    st.rerun() # Rerun the app to clear the displayed messages immediately

# --- Chat Input with UI Feedback ---
# We'll use a placeholder for the chat input text
prompt = st.chat_input("Say something...", disabled=st.session_state.get("thinking", False))
# `disabled` state will be managed when AI is responding.

if prompt:
    # 1. Add user message to chat history and display immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Prepare chat history for FastAPI
    chat_history_for_api = []
    for msg in st.session_state.messages:
        if msg["role"] == "user" or msg["role"] == "assistant":
            chat_history_for_api.append({
                "role": msg["role"],
                "parts": [{"text": msg["content"]}]
            })

    # --- Communicate with FastAPI Backend with Enhanced Error Handling & UI Feedback ---
    try:
        # Set a flag to disable input and show spinner
        st.session_state.thinking = True
        with st.spinner("Gemini is thinking..."):
            response = requests.post(
                CHAT_ENDPOINT,
                json={"message": prompt, "chat_history": chat_history_for_api},
                timeout=60 # Add a timeout to prevent indefinite waiting
            )
            response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)

            # Parse the JSON response
            ai_response_data = response.json()
            ai_message = ai_response_data.get("ai_message", "Sorry, I couldn't get a clear response from Gemini.")

        # 3. Add AI message to chat history and display
        st.session_state.messages.append({"role": "assistant", "content": ai_message})
        with st.chat_message("assistant"):
            st.markdown(ai_message)

    except requests.exceptions.ConnectionError:
        st.error("Connection Error: Could not connect to the backend. Please ensure the FastAPI server is running and accessible.")
    except requests.exceptions.Timeout:
        st.error("Request Timeout: The backend took too long to respond. Please try again.")
    except requests.exceptions.HTTPError as e:
        # More detailed error handling for HTTP responses
        status_code = e.response.status_code
        error_detail = "An unknown error occurred."
        try:
            error_json = e.response.json()
            error_detail = error_json.get("detail", error_detail)
        except json.JSONDecodeError:
            error_detail = e.response.text # Fallback to raw text if not JSON

        if status_code == 400:
            st.warning(f"Bad Request (Code 400): {error_detail}. Your prompt might have been too long or violated safety policies.")
        elif status_code == 404:
            st.error(f"Not Found (Code 404): {error_detail}. The backend endpoint might be incorrect or missing.")
        elif status_code == 500:
            st.error(f"Internal Server Error (Code 500): {error_detail}. The backend encountered a problem.")
        else:
            st.error(f"HTTP Error {status_code}: {error_detail}")
    except json.JSONDecodeError:
        st.error("Invalid Response: Received an unreadable response from the backend. The server might be misconfigured.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
    finally:
        # Always reset the thinking flag
        st.session_state.thinking = False
        # Re-run Streamlit to update the UI (e.g., re-enable input)
        st.rerun() # Ensure input is re-enabled if an error occurs