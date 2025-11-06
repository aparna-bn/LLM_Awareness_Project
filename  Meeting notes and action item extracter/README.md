# Meeting Notes Extractor

This application transcribes audio recordings of meetings and uses an AI model to generate structured notes, including a summary, action items, and key decisions.

## Features

-   **Audio Transcription:** Uses OpenAI's Whisper model to transcribe audio files (MP3, WAV, M4A).
-   **AI-Powered Note Generation:** Leverages the Google Gemini API to analyze the transcript and extract key information.
-   **Structured Output:** Presents the output in a clean, organized format with separate sections for a summary, action items, and key decisions.

## Getting Started

### Prerequisites

-   Python 3.7+
-   pip
-   **FFmpeg:** Whisper requires FFmpeg to be installed on your system. You can install it using your system's package manager:
    -   **Ubuntu/Debian:** `sudo apt update && sudo apt install ffmpeg`
    -   **macOS (using Homebrew):** `brew install ffmpeg`
    -   **Windows (using Chocolatey):** `choco install ffmpeg`

### Installation

1.  **Navigate to the `Meeting notes and action item extracter` directory:**

    ```bash
    cd "LLM_Awareness_Project/Meeting notes and action item extracter"
    ```

2.  **Install the required Python dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

### Configuration

1.  **Create a `.env` file** in the `Meeting notes and action item extracter` directory.

2.  **Add your Gemini API key to the `.env` file:**

    ```
    GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
    ```

    Replace `"YOUR_GEMINI_API_KEY"` with your actual Gemini API key.

### Running the Application

1.  **Run the Streamlit app:**

    ```bash
    streamlit run app_streamlit.py
    ```

2.  The application will open in your web browser. Upload an audio file and click "Process Audio File" to get the meeting notes.
