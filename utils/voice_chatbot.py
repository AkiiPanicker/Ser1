import os
from IPython.display import Audio, display
from utils.audio_recorder import record_audio
from utils.speech_to_text import speech_to_text
from utils.text_to_speech import text_to_speech
import google.generativeai as genai

def run_voice_chatbot(model):
    """
    Interactive voice chatbot loop using Gemini model.
    """
    print("Voice Chatbot Activated. Press Enter to speak or type 'quit'.")

    while True:
        trigger = input()
        if trigger.strip().lower() == 'quit':
            print("Chatbot session ended.")
            break

        print("Listening...")
        audio_file = "user_input_audio.webm"
        record_audio(audio_file)

        text = speech_to_text(audio_file)

        if os.path.exists(audio_file):
            os.remove(audio_file)

        if text:
            print(f"You said: {text}")
            try:
                response = model.generate_content(text)
                print("Bot:", response.text)

                text_to_speech(response.text)
                display(Audio("response.mp3", autoplay=True))

            except Exception as e:
                print(f"Error generating response: {e}")
        else:
            print("Sorry, didn't catch that.")
