# Multimodal Assistant Architecture

The Multimodal Assistant is designed with a clear separation of concerns, with distinct modules for the user interface, chat logic, and the AI pipeline.

## 1. Streamlit User Interface (`app/ui.py`)

This is the main entry point of the application. It uses the Streamlit library to create a web-based user interface with the following components:

-   **Tabs:** The UI is divided into two tabs: one for text-only chat and another for image-based chat.
-   **Chat History:** Each tab maintains its own chat history.
-   **File Uploader:** The image assistant tab includes a file uploader for users to provide an image.

## 2. Chat Session Management (`app/chat.py`)

The `ChatSession` class is responsible for managing the state of a conversation. For the image assistant, it holds the uploaded image in context for a single question-answer exchange.

## 3. AI Pipeline (`app/pipeline.py`)

This module is the bridge between the application and the Google Gemini API. It contains the `process_text_and_image` function, which:

-   Takes a text query and an optional image (as bytes).
-   Uses the `google-generativeai` library to send the data to the `gemini-2.0-flash` model.
-   Returns the model's response as text.

## Architectural Diagram

The following diagram illustrates the application's architecture and data flow:

```mermaid
flowchart TD
    subgraph User Interface (ui.py)
        direction LR
        A[Text Assistant Tab] --> C{ChatSession}
        B[Image Assistant Tab] --> C
    end

    subgraph Backend Logic
        direction TB
        C --> D[Pipeline (pipeline.py)]
    end

    subgraph External Services
        direction TB
        D --> E[Google Gemini API]
    end

    E --> D
    D --> C
    C --> A
    C --> B
```
