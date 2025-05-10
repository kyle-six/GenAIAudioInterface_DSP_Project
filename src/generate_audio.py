import torchaudio
from tangoflux import TangoFluxInference
from pydub import AudioSegment
from pydub.playback import play
import numpy as np
import os

class TangoFluxGenerator:
    def __init__(self, model_name='declare-lab/TangoFlux', cache_dir='models'):
        self.model = None
        self.model_name = model_name
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        
def load_model(self):
    # Load the TangoFlux model
    if self.model is None:
        self.model = TangoFluxInference(name=self.model_name, cache_dir=self.cache_dir)
        print(f"Model loaded on: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        

def generate(self, prompt, steps=50, duration=10.0):
    # Generate audio from text
    self.load_model()
    audio_tensor = self.model.generate(prompt, steps=steps, duration=duration)
    return audio_tensor

@staticmethod
def play_audio(audio_tensor, sample_rate=44100):
    # Play audio directly from tensor
    audio_np = audio_tensor.numpy()
    audio_segment = AudioSegment(
        audio_np.tobytes(),
        frame_rate=sample_rate,
        sample_width=audio_np.dtype.itemsize,
        channels=1
    )
    play(audio_segment)

@staticmethod
def save_wav(audio_tensor, filepath, sample_rate=44100):
    # Save to WAV file
    torchaudio.save(filepath, audio_tensor, sample_rate)        
        