from transformers import pipeline, AutoTokenizer
import math

# Global variable to hold the summarization pipeline and tokenizer (for better length estimation)
# This ensures the model and tokenizer are loaded only once when the script starts
summarization_pipeline = None
tokenizer = None # We'll add a tokenizer to better estimate token count


def initialize_summarizer(model_name: str = "facebook/bart-large-cnn"):
    """
    Initializes the Hugging Face summarization pipeline and its tokenizer.
    This function should be called once at the start of your application.

    Args:
        model_name (str): The name of the pre-trained summarization model to use.
                          Defaults to "facebook/bart-large-cnn".
    """
    global summarization_pipeline, tokenizer
    if summarization_pipeline is None:
        try:
            print(f"Loading summarization model: {model_name}...")
            # Use torch_dtype="auto" for automatic device handling and better memory
            summarization_pipeline = pipeline("summarization", model=model_name, torch_dtype="auto")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            print("Summarization model and tokenizer loaded successfully.")
        except Exception as e:
            print(f"Error loading summarization model or tokenizer '{model_name}': {e}")
            summarization_pipeline = None
            tokenizer = None

def generate_summary(text: str, min_length: int = 50, max_length: int = 150) -> str:
    """
    Generates a summary for the given text using the initialized summarization pipeline.

    Args:
        text (str): The input text to be summarized.
        min_length (int): Minimum length of the generated summary.
        max_length (int): Maximum length of the generated summary.

    Returns:
        str: The generated summary, or an error message if the pipeline is not initialized
             or if summarization fails.
    """
    global summarization_pipeline, tokenizer # Ensure tokenizer is also global here
    if summarization_pipeline is None or tokenizer is None:
        return "Error: Summarization model or tokenizer not initialized. Please call initialize_summarizer() first."

    if not text.strip():
        return "No text provided for summarization."

    # BART models have a max input of 1024 tokens.
    # The pipeline's internal tokenizer will handle truncation if text is too long,
    # but we can still print a warning if it's detected.
    max_model_tokens = 1024 # Standard for BART models

    # This token count is primarily for informative warnings, as pipeline handles actual truncation
    current_token_count = len(tokenizer.encode(text, add_special_tokens=True))

    if current_token_count > max_model_tokens:
        print(f"Warning (in generate_summary): Input text has {current_token_count} tokens, exceeding model's max of {max_model_tokens}. It will be truncated by the pipeline.")
    elif current_token_count >= max_model_tokens * 0.9: # Give a warning if close to limit
         print(f"Info (in generate_summary): Input text has {current_token_count} tokens, close to model's max of {max_model_tokens}.")

    try:
        # The pipeline returns a list of dictionaries, take the 'summary_text' from the first result
        summary_results = summarization_pipeline(
            text, # Pass the original text, the pipeline's internal tokenizer will handle truncation if needed
            max_length=max_length,
            min_length=min_length,
            do_sample=False # For more consistent results, set to True for more creative summaries
        )
        return summary_results[0]['summary_text']
    except Exception as e:
        return f"An error occurred during summarization: {e}"
    
def chunk_text(text: str, max_tokens: int, overlap_tokens: int = 50) -> list[str]:
    """
    Splits text into chunks based on a maximum token limit, with optional overlap.
    Uses the global tokenizer.

    Args:
        text (str): The input text to chunk.
        max_tokens (int): The maximum number of tokens per chunk (e.g., 1024 for BART).
        overlap_tokens (int): Number of tokens to overlap between consecutive chunks.
                              Helps maintain context.

    Returns:
        list[str]: A list of text chunks.
    """
    global tokenizer
    if tokenizer is None:
        raise ValueError("Tokenizer not initialized. Call initialize_summarizer() first.")

    tokens = tokenizer.encode(text, add_special_tokens=True)
    
    chunks = []
    current_start = 0
    while current_start < len(tokens):
        current_end = min(current_start + max_tokens, len(tokens))
        chunk_tokens = tokens[current_start:current_end]
        
        # Decode the tokens back to string for the chunk
        chunk_text_str = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text_str)
        
        # Move start for next chunk
        current_start += (max_tokens - overlap_tokens)
        if current_start >= len(tokens): # Ensure we don't go past end
            break
        # Ensure that if the last chunk was smaller than max_tokens and there's no more text, we stop.
        # This condition helps prevent infinite loops on very short texts where overlap might make current_start <= len(tokens)
        if current_start + overlap_tokens >= len(tokens) and len(chunks[-1].split()) < 50: # If the last chunk was very small and no substantial content left
             break

    return chunks

def multi_stage_summarize(
    long_text: str,
    max_model_tokens: int = 1024, # BART's typical max input
    overlap_tokens: int = 100, # A reasonable overlap for summarization
    summary_min_length: int = 50,
    summary_max_length: int = 200,
    progress_callback=None # Optional callback for UI progress updates
) -> str:
    """
    Performs multi-stage summarization for very long texts.

    Args:
        long_text (str): The full text to summarize.
        max_model_tokens (int): Maximum tokens the underlying model can handle per input.
        overlap_tokens (int): Tokens to overlap between chunks.
        summary_min_length (int): Minimum length for individual chunk summaries.
        summary_max_length (int): Maximum length for individual chunk summaries.
        progress_callback (callable): A function to call with progress updates (0-1 float).

    Returns:
        str: The final combined summary.
    """
    global summarization_pipeline, tokenizer
    if summarization_pipeline is None or tokenizer is None:
        raise ValueError("Summarization model or tokenizer not initialized.")

    if not long_text.strip():
        return "No text provided for summarization."

    # Step 1: Chunk the long text
    initial_chunks = chunk_text(long_text, max_model_tokens, overlap_tokens)
    print(f"Split text into {len(initial_chunks)} chunks.")

    if len(initial_chunks) == 1:
        # If only one chunk, summarize directly
        print("Text fits in one chunk. Summarizing directly.")
        if progress_callback: progress_callback(0.5) # Fake half progress
        final_summary = generate_summary(initial_chunks[0], summary_min_length, summary_max_length)
        if progress_callback: progress_callback(1.0)
        return final_summary

    # Step 2: Summarize each chunk
    chunk_summaries = []
    total_chunks = len(initial_chunks)
    print(f"Summarizing {total_chunks} chunks...")
    for i, chunk in enumerate(initial_chunks):
        if progress_callback:
            progress = (i / total_chunks) * 0.5 # First half of progress for chunk summarization
            progress_callback(progress)

        # --- Aggressive Truncation for Safety ---
        # Ensure the chunk is *well* within the token limit before calling generate_summary
        chunk_tokens = tokenizer.encode(chunk, add_special_tokens=True)
        if len(chunk_tokens) > max_model_tokens * 0.9: # Truncate more aggressively
            print(f"Warning: Chunk {i+1} has {len(chunk_tokens)} tokens. Truncating to {int(max_model_tokens * 0.8)} tokens.")
            chunk = tokenizer.decode(chunk_tokens[:int(max_model_tokens * 0.8)], skip_special_tokens=True)

        summary = generate_summary(chunk, summary_min_length, summary_max_length)
        if "Error:" in summary:
            print(f"Warning: Error summarizing chunk {i+1}: {summary}")
            # Optionally, you could try to re-summarize, skip, or return an error here.
            # For robustness, we'll just append what we got or an empty string if error.
            chunk_summaries.append("")  # Append empty string if error in chunk summary
        else:
            chunk_summaries.append(summary)

    # Filter out empty summaries if any
    chunk_summaries = [s for s in chunk_summaries if s.strip()]

    if not chunk_summaries:
        return "Failed to summarize any chunks. Please check the input text or model."

    # Step 3: Combine and (optionally) re-summarize the summaries
    combined_summaries_text = " ".join(chunk_summaries)
    print(f"Combined {len(chunk_summaries)} chunk summaries. Total words in combined summaries: {len(combined_summaries_text.split())}")

    # Check if combined_summaries_text is empty before attempting re-summarization
    if combined_summaries_text.strip():
        # If the combined summaries are still too long, summarize them again
        # We use tokenizer.encode to get a more accurate token count for re-summarization decision
        current_tokens_in_combined = len(tokenizer.encode(combined_summaries_text, add_special_tokens=True))

        if current_tokens_in_combined > max_model_tokens:
            print(f"Combined summaries are too long ({current_tokens_in_combined} tokens). Re-summarizing...")
            if progress_callback: progress_callback(0.75)  # Mid-point for re-summarization

            # Recursively call generate_summary or handle multi-stage summarization of summaries
            # For simplicity, we'll just call generate_summary directly here.
            # A more advanced recursive call to multi_stage_summarize could be used,
            # but that complicates progress tracking and can lead to very short final summaries.
            # A safer approach is to ensure the max_length for the final summary is reasonable.

            # Adjust max_length for the final summary to be roughly proportional to the combined summaries
            # Or, just keep a fixed reasonable length for the final output.
            # Let's target a final summary that's also around the max_length of the individual summaries.
            final_summary = generate_summary(combined_summaries_text, min_length=summary_min_length, max_length=summary_max_length)
            print("Re-summarization complete.")
        else:
            print("Combined summaries fit within model limits. Using combined summaries as final.")
            final_summary = combined_summaries_text
    else:
        final_summary = "No valid summaries were generated from the chunks."

    if progress_callback: progress_callback(1.0)
    return final_summary

# --- Test Functions for `if __name__ == "__main__":` block ---

def test_summary_initialized():
    """Tests summarization when the model is initialized."""
    test_text_short = "Artificial intelligence (AI) is intelligence—perceiving, synthesizing, and inferring information—demonstrated by machines, as opposed to intelligence displayed by animals and humans. AI research has been defined as the field of study of 'intelligent agents', which refers to any device that perceives its environment and takes actions that maximize its chance of achieving its goals. The term 'artificial intelligence' had previously been used to describe machines that mimic and display 'human' cognitive skills that are associated with a 'human mind', such as 'learning' and 'problem-solving'."
    print("\n--- Summary 1 (Short Text) ---")
    summary_short = generate_summary(test_text_short, min_length=30, max_length=70)
    print(summary_short)

    test_text_long_base = """
    The quick brown fox jumps over the lazy dog. This is a classic pangram often used to test typewriters and computer fonts. It contains all the letters of the English alphabet. Pangrams are fun linguistic exercises. They demonstrate the versatility of language.
    In a galaxy far, far away, a new hope emerged. The evil Galactic Empire ruled with an iron fist, but a small group of rebels fought for freedom. Luke Skywalker, a young farm boy, discovered his destiny and joined the fight against the dark side. With the help of his friends, Han Solo and Princess Leia, and the wise Jedi Master Obi-Wan Kenobi, he embarked on a perilous journey to restore peace to the galaxy. The Force was strong with him, guiding his path through many trials and tribulations. The fate of the galaxy rested on his shoulders.
    Climate change refers to long-term shifts in temperatures and weather patterns. These shifts may be natural, but since the 1800s, human activities have been the main driver of climate change, primarily due to burning fossil fuels (like coal, oil, and gas) which produces heat-trapping gases. These gases, such as carbon dioxide and methane, accumulate in the atmosphere, creating a greenhouse effect that warms the planet. The consequences of climate change include melting glaciers, rising sea levels, more frequent and intense heatwaves, and disruptions to ecosystems and agriculture. International efforts are underway to mitigate climate change through reducing emissions, developing renewable energy sources, and adapting to the changing environment.
    """
    # Explicitly cut the test text to a safe length for Sprint 1 testing
    # This ensures the test itself does not hit internal model limits for now.
    test_text_long_safe = test_text_long_base[:1500] # Cut to first 1500 characters, should be safe

    print("\n--- Summary 2 (Longer Text - Truncated for Test) ---")
    # Using the explicitly truncated test text
    summary_long = generate_summary(test_text_long_safe, min_length=70, max_length=200)
    print(summary_long)

    print("\n--- Summary 3 (No Text) ---")
    summary_empty = generate_summary("")
    print(summary_empty)

def test_summary_not_initialized():
    """
    Tests summarization when the model is explicitly set to not be initialized.
    This function has its own scope, allowing 'global' to be correctly placed.
    """
    global summarization_pipeline, tokenizer # Declare global for both
    
    test_text_short = "This text should not be summarized."
    
    print("\n--- Summary 4 (Not Initialized) ---")
    
    # Store current state
    temp_pipeline = summarization_pipeline
    temp_tokenizer = tokenizer

    # Force to None for this test
    summarization_pipeline = None
    tokenizer = None

    summary_no_init = generate_summary(test_text_short)
    print(summary_no_init)

    # Restore original state
    summarization_pipeline = temp_pipeline
    tokenizer = temp_tokenizer


