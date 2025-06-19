from gtts import gTTS
from pydub import AudioSegment

def text_to_speech(text, filename='response.mp3', speed_factor=1.33):
    """
    Converts text to speech, speeds up, and saves as an MP3.
    """
    try:
        tts = gTTS(text=text, lang='en')
        tts.save(filename)

        audio = AudioSegment.from_mp3(filename)
        sped_up = audio.set_frame_rate(int(audio.frame_rate * speed_factor))
        sped_up.export(filename, format="mp3")

        print(f"Generated and sped up audio saved as {filename}")

    except Exception as e:
        print(f"TTS error: {e}")
