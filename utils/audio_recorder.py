import pyaudio
import wave

def record_audio_pyaudio(filename='audio.wav', chunk=1024, sample_format=pyaudio.paInt16, channels=2, fs=44100, seconds=5):
    """
    Records audio from the microphone using PyAudio and saves it as a WAV file.

    Args:
        filename (str): The name of the output WAV file.
        chunk (int): The number of frames per buffer.
        sample_format (int): The audio format (e.g., paInt16 for 16-bit integers).
        channels (int): The number of audio channels (1 for mono, 2 for stereo).
        fs (int): The sample rate.
        seconds (int): The duration of the recording in seconds.
    """
    p = pyaudio.PyAudio()

    print(f"Recording for {seconds} seconds...")

    try:
        stream = p.open(format=sample_format,
                        channels=channels,
                        rate=fs,
                        frames_per_buffer=chunk,
                        input=True)

        frames = []
        for _ in range(0, int(fs / chunk * seconds)):
            data = stream.read(chunk)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b''.join(frames))
        wf.close()

        print(f"Recording stopped. Audio saved as {filename}")

    except Exception as e:
        print(f"An error occurred during recording: {e}")
        print("Please ensure PortAudio is installed and your microphone is accessible.")

if __name__ == '__main__':
    record_audio_pyaudio()
