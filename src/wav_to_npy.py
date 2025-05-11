import librosa
import numpy as np

file_name = 'src/impulse_responses/large_hall.wav'

# Load IR as mono (default) and normalize
ir, sr = librosa.load(file_name, sr=None)  
ir = ir / np.max(np.abs(ir))  # Normalize

# Save as .npy
np.save('src/impulse_responses/large_hall.npy', ir)
