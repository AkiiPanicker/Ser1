import speech_recognition as sr
from pydub import AudioSegment
import os

def speech_to_text(audio_file_path):
    """
    Converts speech from a .webm audio file to text.
    """
    try:
        audio = AudioSegment.from_file(audio_file_path, format="webm")
        converted = "converted_audio.wav"
        audio.export(converted, format="wav")

        r = sr.Recognizer()
        with sr.AudioFile(converted) as source:
            audio_data = r.record(source)

        os.remove(converted)
        return r.recognize_google(audio_data)

    except sr.UnknownValueError:
        print("Speech Recognition could not understand audio")
        return None
    except sr.RequestError as e:
        print(f"Google Speech Recognition error: {e}")
        return None
    except Exception as e:
        print(f"General error: {e}")
        return None
