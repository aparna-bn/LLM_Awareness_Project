import os
import json
import google.generativeai as genai

def generate_meeting_notes(transcript):
    # ... (No changes here, it uses GEMINI_API_KEY as planned)
    # ... (Uses gemini-flash-latest for the summary/extraction)
    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel('gemini-flash-latest')

        prompt = f"""
        You are a professional meeting assistant specializing in extracting structured notes.
        Please analyze the following meeting transcript. Your output should be a JSON object containing three keys:
        1.  "summary": A concise overview of the key discussion points and outcomes.
        2.  "action_items": A list of specific tasks. **EACH ITEM MUST BE A SINGLE STRING** formatted as 'Owner: Task description with deadline.'
        3.  "key_decisions": A list of all final decisions or agreements made during the meeting.

        Ensure that the format is valid JSON.

        Here is the transcript:
        ---
        {transcript}
        ---
        """
        
        response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        
        return json.loads(response.text)
    except Exception as e:
        print(f"An error occurred during note generation: {e}")
        return None