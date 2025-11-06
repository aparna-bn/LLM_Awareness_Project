# Global News Tracker

This application is a desktop tool that fetches news from various sources, groups them into topics, and uses an AI model to generate summaries for each topic. The results are displayed in a user-friendly graphical interface.

## Features

-   **RSS Feed Aggregation:** Fetches news from multiple Google News RSS feeds.
-   **Topic Clustering:** Automatically groups related articles into topics using TF-IDF and cosine similarity.
-   **AI-Powered Summarization:** Uses the Google Gemini API to generate concise summaries of the news topics.
-   **GUI Dashboard:** A Tkinter-based graphical user interface to display the trending topics and their summaries.

## Getting Started

### Prerequisites

-   Python 3.7+
-   pip

### Installation

1.  **Navigate to the `Global_News_Tracker` directory:**

    ```bash
    cd LLM_Awareness_Project/Global_News_Tracker
    ```

2.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

### Configuration

1.  **Create a `.env` file** in the `Global_News_Tracker` directory.

2.  **Add your Gemini API key to the `.env` file:**

    ```
    GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
    ```

    Replace `"YOUR_GEMINI_API_KEY"` with your actual Gemini API key.

### Running the Application

1.  **Run the application:**

    ```bash
    python global_news_tracker.py
    ```

2.  The application will open a GUI window. Click the "Get News Summary" button to start the news fetching and processing pipeline.
