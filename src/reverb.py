# Real-time convolution of audio with a reverb IR using the FFT-based overlap-save method
import numpy as np
from scipy.fft import fft, ifft

def init_reverb(ir_path: str, block_size: int) -> tuple: # Load pre-computed impulse response (.npy)      
    h = np.load(ir_path)  
    h = h / np.max(np.abs(h))  # Normalize to prevent clipping
    
    buffer = np.zeros(len(h) - 1 + block_size) # Create buffer for overlap-save
    H = fft(np.pad(h, (0, len(buffer) - len(h)))) # Zero-pad IR, compute FFT
    return H, buffer

def apply_reverb(x_block: np.ndarray, H: np.ndarray, buffer: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    
    N_ir = len(buffer) - len(x_block) + 1   # Calculate len of IR
    
    # Update buffer (overlap-save)
    buffer[:N_ir-1] = buffer[-N_ir+1:]
    buffer[N_ir-1:] = x_block
    
    # Convolve
    X = fft(buffer) # Compute FFT of the buffer
    y_full = ifft(X * H).real # Perform freq-domain conv, take IFFT
    
    return y_full[-len(x_block):], buffer.copy() # Return overlap-save output
