# uvicorn fastapi_app2:app --reload --port 8001
import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

import google.generativeai as genai
import logging
# NEW IMPORT: Import the correct exception type
from google.api_core.exceptions import GoogleAPIError 

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
GEMINI_API_KEYS = os.getenv("GEMINI_API_KEYS")
if not GEMINI_API_KEYS:
    logger.error("GEMINI_API_KEYS not found in environment variables. Please set it in your .env file.")
    raise ValueError("GEMINI_API_KEYS not found in environment variables. Please set it in your .env file.")

# Configure the Gemini API
genai.configure(api_key=GEMINI_API_KEYS)

# Initialize the Gemini model
model = None
try:
    # IMPORTANT: Ensure this is a valid model name (e.g., 'gemini-1.5-flash', not 'gemini-2.0-flash')
    model = genai.GenerativeModel('gemini-flash-latest')
    logger.info("Gemini model 'gemini-flash-latest' initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing Gemini model 'gemini-flash-latest': {e}")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="GeminiSpeak Backend",
    description="FastAPI backend for Real-Time Gemini Chat Assistant"
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models for Request/Response ---
class ChatPart(BaseModel):
    text: str = Field(..., description="The text content of a message part.")

class ChatMessage(BaseModel):
    role: str = Field(..., description="The role of the message sender (e.g., 'user', 'model').")
    parts: List[ChatPart] = Field(..., description="A list of content parts, typically text.")

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000, description="The user's new message.")
    chat_history: List[ChatMessage] = Field(..., description="The full conversation history for context.")

class ChatResponse(BaseModel):
    ai_message: str = Field(..., description="The AI's generated response.")

# --- API Endpoints ---
@app.post("/chat", response_model=ChatResponse)
async def chat_with_gemini(request: ChatRequest):
    if not model:
        logger.error("Attempted to call chat endpoint but Gemini model was not initialized.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Gemini model not initialized. Backend service unavailable."
        )

    if not request.message.strip():
        logger.warning(f"Received empty message from frontend.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message cannot be empty."
        )

    try:
        formatted_history = []
        for msg in request.chat_history:
            formatted_parts = [{"text": part.text} for part in msg.parts]
            formatted_history.append({"role": msg.role, "parts": formatted_parts})

        logger.info(f"Received message: '{request.message}' with history length: {len(formatted_history)}")

        convo = model.start_chat(history=formatted_history)
        response = await convo.send_message_async(request.message)

        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            ai_text = response.candidates[0].content.parts[0].text
        else:
            ai_text = "Sorry, I couldn't generate a coherent response."
            logger.warning(f"Gemini returned an empty or malformed response for message: '{request.message}'")

        logger.info(f"Gemini responded: '{ai_text[:50]}...'")
        return ChatResponse(ai_message=ai_text)
    
    # Catching the specific BlockedPromptException first is still a good practice
    except genai.types.BlockedPromptException as e:
        logger.warning(f"Prompt blocked by safety settings: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Your prompt was blocked by safety settings. Please try rephrasing: {e}"
        )
    # NEW: Catch the more general GoogleAPIError
    except GoogleAPIError as e:
        logger.error(f"Gemini API Error: {e.args[0]}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gemini API error: {e.args[0]}"
        )
    except Exception as e:
        logger.error(f"Unexpected error in chat_with_gemini: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected internal server error occurred: {e}"
        )

@app.get("/")
async def root():
    return {"message": "Welcome to GeminiSpeak Backend! Visit /docs for API documentation."}