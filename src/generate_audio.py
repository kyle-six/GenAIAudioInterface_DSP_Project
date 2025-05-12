#import torchaudio
import os, torch, torchaudio, librosa, time
import numpy as np
from model_interface import ModelInterface
from tangoflux import TangoFluxInference
from pathlib import Path
from diffusers import AudioLDMPipeline



class TangoFluxModel(ModelInterface):    
    def __init__(self):
        super().__init__()
    
    def load(self, path: str = ""):
        if self.model != None:
            print("Error: Model already loaded")
            return
        
        # Use Pre-Trained model
        if path == "" or not Path(path).exists():
            print("Checkpoint file not found, loading pre-trained model...")
            print("################# TangoFlux Loading #################")
            self.model = TangoFluxInference(name='declare-lab/TangoFlux')
            print("Model loaded into memory!")
        else:
            try:
                pass
                # TODO: load custom tangoflux
                #self.model = build_model(ckpt_path=path, model_name="audioldm-s-text-ft")
            except:
                raise ValueError("Incorrect checkpoint path to load AudioLDM model.")  
        
    def infer(self, prompt: str = "static", duration: float = 10.0, steps: int = 50):
        audio = None
        if self.model == None:
            print("Error: Model not loaded")
            return audio
        print("################# Generating Audio #################")
        print(f"Prompt: {prompt}")
        print(f"Duration: {duration}")
        print(f"Running inference on {('gpu' if torch.cuda.is_available() else 'cpu')}...")
        audio = self.model.generate(prompt, steps=steps, duration=duration)
        
        #saved_audio_path = f"{time.strftime('%Y%m%d-%H%M%S')}.wav"
        #torchaudio.save(saved_audio_path, src=audio, sample_rate=44100)
        
        return audio


class AudioldmModel(ModelInterface):
    DEFAULT_REPO_ID = "cvssp/audioldm-s-full-v2"
    
    INFERENCE_STEPS = 10
    
    #GEN_DURATION_SEC = 5.0
    #TARGET_DURATION_SEC = 4.0
    SAMPLE_RATE = 44100
    HOP_DURATION_SEC = 0.1

    def __init__(self):
        super().__init__()
        self.model = None

    def load(self, path: str = ""):
        if self.model is not None:
            print("Warning: AudioLDM model already loaded.")
            return
        
        if path == "" or not Path(path).exists():
            print("Checkpoint file not found, loading pre-trained model...")
            print("################# AudioLDM Loading #################")
            self.model = AudioLDMPipeline.from_pretrained(self.DEFAULT_REPO_ID, torch_dtype=torch.float16)
        else:
            try:
                self.model = AudioLDMPipeline.from_pretrained(path, torch_dtype=torch.float16)
            except:
                raise ValueError("Incorrect checkpoint path to load AudioLDM model.")  
            
        # Move to correct GPU/CPU
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        print("AudioLDM model loaded.")

    def infer(self, prompt: str, duration: float, steps: int):
        if self.model is None:
            print("Error: Model not loaded.")
            return None
        print("################# Generating Audio #################")
        print(f"Prompt: {prompt}")
        print(f"Duration: {duration}")
        print(f"Running inference on {'cuda' if torch.cuda.is_available() else 'cpu'}...")
        waveform = self.model(prompt, num_inference_steps=steps, audio_length_in_s=duration + 1.0).audios[0]

        # # Extract loudest 4 seconds of audio
        # cropped_waveform = self.find_loudest_segment(waveform, sr=self.SAMPLE_RATE, segment_length=int(duration), hop_length_sec=self.HOP_DURATION_SEC)
        # cropped_waveform = torch.tensor(cropped_waveform, dtype=torch.float32).unsqueeze(0)
        #saved_audio_path = f"{time.strftime('%Y%m%d-%H%M%S')}.wav"
        #torchaudio.save(saved_audio_path, src=waveform, sample_rate=44100)
        
        return waveform
    
    @staticmethod
    def find_loudest_segment(audio: np.ndarray, sr: int, segment_length: int = 4, hop_length_sec: float = 1.0) -> np.ndarray:
        hop_length_samples = int(sr * hop_length_sec)
        frame_length_samples = int(sr * segment_length)

        if len(audio) < frame_length_samples:
            return np.pad(audio, (0, frame_length_samples - len(audio)), mode='constant')

        rms_values = librosa.feature.rms(
            y=audio,
            frame_length=frame_length_samples,
            hop_length=hop_length_samples,
            center=False
        ).squeeze()

        max_rms_index = np.argmax(rms_values)
        start_sample = max_rms_index * hop_length_samples
        end_sample = start_sample + frame_length_samples

        return audio[start_sample:end_sample]

