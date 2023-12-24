import pyaudio
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import aubio
from termcolor import colored  # Install the 'termcolor' library for colored text


# Function to calculate volume
def calculate_volume(audio_data):
    return np.mean(librosa.feature.rms(y=audio_data))

# Function to calculate pitch
def calculate_pitch(audio_data):
    pitch_o = aubio.pitch(method="yin", buf_size=1024, hop_size=1024)
    pitch_o.set_unit("midi")
    pitch_o.set_tolerance(0.8)
    return pitch_o(audio_data)[0]

# Function to calculate tempo
def calculate_tempo(audio_data, sr):
    tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sr)
    return tempo

# Function to calculate spectral centroid
def calculate_spectral_centroid(audio_data, sr):
    # Calculate the required padding based on n_fft
    n_fft=2048
    signal_length = len(audio_data)
    needed_padding = n_fft - (signal_length % n_fft)

    # Apply zero-padding to the input signal
    padded_audio = np.pad(audio_data, (0, needed_padding), 'constant')

    # Calculate spectral centroid using the padded signal
    spectral_centroid = librosa.feature.spectral_centroid(y=padded_audio, sr=sr)[0]

    return spectral_centroid

def interpret_results(volume, pitch, tempo, spectral_centroid):
    # Map volume to colors
    volume_color = 'red' if volume > 0.5 else 'green'

    # Map pitch to colors
    pitch_color = 'magenta' if pitch > 60 else 'cyan'

    # Map tempo to colors
    tempo_color = 'yellow' if tempo > 120 else 'blue'

    # Map spectral centroid to colors
    centroid_color = 'white' if spectral_centroid.any() > 2000 else 'grey'

    # Combine colors
    overall_color = f"Volume: {colored('■', volume_color)} | Pitch: {colored('■', pitch_color)} | Tempo: {colored('■', tempo_color)} | Spectral Centroid: {colored('■', centroid_color)}"

    return overall_color


# Audio parameters
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100
CHUNK = 1024

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

try:
    while True:
        # Read audio data from the stream
        audio_data = np.frombuffer(stream.read(CHUNK), dtype=np.float32)

        # Calculate features
        volume = calculate_volume(audio_data)
        pitch = calculate_pitch(audio_data)
        tempo = calculate_tempo(audio_data, RATE)
        spectral_centroid = calculate_spectral_centroid(audio_data, RATE)

        # Interpret results and print colored output
        colored_output = interpret_results(volume, pitch, tempo, spectral_centroid)
        print(colored_output)

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    # Close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()