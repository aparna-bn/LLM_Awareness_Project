import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# app/test_query.py
from pipeline import start_image_chat_session, ask_followup

image_path = "/home/unovie/Downloads/LLM-Awareness-Project/Multimodal_assisstant/app/example.png"

# Start session with image
chat, image = start_image_chat_session(image_path)
if chat is None:
    print("Failed to start session:", image)
    exit()

# Initial question with image
response = ask_followup(chat, "What is shown in the image?", image)
print("Initial Response:", response)

# Follow-up 1
follow_up_1 = ask_followup(chat, "What season does this image likely represent?")
print("Follow-up 1:", follow_up_1)

# Follow-up 2
follow_up_2 = ask_followup(chat, "Can you describe the mood or feeling of the scene?")
print("Follow-up 2:", follow_up_2)
