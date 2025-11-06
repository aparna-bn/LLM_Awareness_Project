# Document Q&A System with Local RAG

This project is a self-contained, local-first Document Q&A system. It uses a Retrieval-Augmented Generation (RAG) architecture to answer questions based on user-uploaded documents. All models (for embeddings and language generation) run locally on your machine.

## Features

-   **Local First:** No reliance on external APIs for the core AI processing.
-   **Document Support:** Upload and process PDF, TXT, and DOCX files.
-   **RAG Pipeline:** Uses LangChain to orchestrate a robust RAG pipeline.
-   **Vector Storage:** Employs ChromaDB for efficient local storage and retrieval of document embeddings.
-   **Caching:** Caches responses to previously asked questions for faster subsequent answers.

## Models Used

-   **Embedding Model:** `sentence-transformers/all-MiniLM-L12-v2`
-   **Language Model:** `google/flan-t5-small`

## Getting Started

### Prerequisites

-   Python 3.8+
-   pip

### Installation

1.  **Navigate to the `Ragbot3` directory:**

    ```bash
    cd LLM_Awareness_Project/Ragbot3
    ```

2.  **Install the required Python dependencies:**

    ```bash
    pip install -r requirements.txt
    ```
    *Note: The first run will download the local models, which may take some time and require a stable internet connection.*

### Running the Application

1.  **Run the Streamlit app:**

    ```bash
    streamlit run main.py
    ```

2.  The application will open in your web browser. Upload your documents, click "Process Documents", and then ask your questions.
