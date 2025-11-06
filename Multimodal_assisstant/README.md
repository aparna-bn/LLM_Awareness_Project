# Multimodal Assistant

This project is a multimodal assistant that can understand and respond to both text and image inputs. It uses Google's Gemini 2.0 Flash model and is built with a Streamlit web interface.

## Features

-   **Text Assistant:** A chat interface for text-based conversations with the AI.
-   **Image Assistant:** An interface to upload an image and ask questions about it.
-   **Multimodal Capabilities:** The assistant can process and understand information from both text and images.

## Getting Started

### Prerequisites

-   Python 3.7+
-   pip

### Installation

1.  **Navigate to the `Multimodal_assisstant` directory:**

    ```bash
    cd LLM_Awareness_Project/Multimodal_assisstant
    ```

2.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

### Configuration

1.  **Create a `.env` file** in the `Multimodal_assisstant` directory.

2.  **Add your Gemini API key to the `.env` file:**

    ```
    GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
    ```

    Replace `"YOUR_GEMINI_API_KEY"` with your actual Gemini API key.

### Running the Application

1.  **Run the Streamlit app from the `Multimodal_assisstant` directory:**

    ```bash
    streamlit run app/ui.py
    ```

2.  The application will open in your web browser.
