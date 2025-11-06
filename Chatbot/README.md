# GeminiSpeak: A Real-Time AI Chat Assistant

This project is a real-time chat application that uses Google's Gemini Pro model to provide intelligent and conversational responses. The application is built with a FastAPI backend and a Streamlit frontend.

## Features

- **Real-Time Chat:** Engage in a conversation with the Gemini AI model.
- **Chat History:** The application maintains the context of the conversation.
- **Clear Chat:** A button to clear the chat history and start a new conversation.
- **Error Handling:** The application handles various errors, such as connection issues and API errors.

## Getting Started

### Prerequisites

- Python 3.7+
- pip

### Installation

1.  **Clone the repository or download the project files.**

2.  **Navigate to the `Chatbot` directory:**
    ```bash
    cd LLM_Awareness_Project/Chatbot
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Configuration

1.  **Create a `.env` file** in the `Chatbot` directory.

2.  **Add your Gemini API key to the `.env` file:**
    ```
    GEMINI_API_KEYS="YOUR_GEMINI_API_KEY"
    ```
    Replace `"YOUR_GEMINI_API_KEY"` with your actual Gemini API key.

### Running the Application

You need to run the backend and the frontend in two separate terminals.

1.  **Run the FastAPI Backend:**
    Open a terminal and run the following command from the `Chatbot` directory:
    ```bash
    uvicorn fastapi_app2:app --reload --port 8001
    ```

2.  **Run the Streamlit Frontend:**
    Open another terminal and run the following command from the `Chatbot` directory:
    ```bash
    streamlit run streamlit_app2.py
    ```

    The application will be available at `http://localhost:8501`.
