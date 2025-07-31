import sounddevice as sd
from scipy.io.wavfile import write
import os

def record_audio(filename="output.wav", duration=10, fs=16000):
    print(f"Recording for {duration} seconds...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, recording)
    print(f"Saved recording to {filename}")
    return filename
