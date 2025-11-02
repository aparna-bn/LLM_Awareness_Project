import whisper
import os

# Load the base model once when the script starts. 
# NOTE: This will download the model (~150MB) on the first run.
try:
    TRANSCRIPTION_MODEL = whisper.load_model("base")
except Exception as e:
    # Handle the case where torch or ffmpeg is not installed
    print(f"Could not load local Whisper model. Check FFmpeg and PyTorch installation: {e}")
    TRANSCRIPTION_MODEL = None

def transcribe_audio(file_path):
    """
    Transcribes an audio file using the local open-source Whisper model.
    """
    if not os.path.exists(file_path):
        return None
    
    if TRANSCRIPTION_MODEL is None:
        return None

    try:
        # Perform transcription using the loaded model
        # You can specify language="en" for slightly faster processing if your audio is only English
        result = TRANSCRIPTION_MODEL.transcribe(file_path)
        
        # The result object contains the transcription text
        return result["text"]
    except Exception as e:
        print(f"Local Transcription Error: {e}")
        return None