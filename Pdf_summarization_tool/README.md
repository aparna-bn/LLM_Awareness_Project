# AI-Powered PDF Summarizer

This project is a web-based application that uses a sophisticated AI model to generate concise summaries of PDF documents. It is designed to handle large documents by breaking them down into smaller chunks and summarizing them in stages.

## Features

-   **PDF Text Extraction:** Extracts text from uploaded PDF files.
-   **Multi-Stage Summarization:** Capable of summarizing large documents that exceed the model's token limit.
-   **Configurable Summary Length:** Allows users to set the minimum and maximum length of the generated summary.
-   **User-Friendly Interface:** A simple and intuitive web interface built with Gradio.

## Getting Started

### Prerequisites

-   Python 3.7+
-   pip

### Installation

1.  **Navigate to the `Pdf_summarization_tool` directory:**

    ```bash
    cd LLM_Awareness_Project/Pdf_summarization_tool
    ```

2.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

1.  **Run the Gradio app:**

    ```bash
    python app.py
    ```

2.  **Open your web browser** and go to the local URL provided by Gradio (usually `http://127.0.0.1:7860`).
