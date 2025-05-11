# Apply distortion to an audio block 

import numpy as np
import torch

def apply_distortion(
    x: np.ndarray,
    mode: str = "soft",
    amount: float = 5.0,
    mix: float = 1.0
) -> np.ndarray:
    """
    Real-time distortion processor with dry/wet mix.
    
    Args:
        x: Input audio (normalized to [-1, 1])
        mode: 'soft', 'hard', 'sine', 'bitcrush'
        amount: Distortion intensity (0.1-10.0)
        mix: Dry/wet ratio (0.0-1.0)
    
    Returns:
        Distorted audio
    """
    x = x.copy()
    dry = x
    
    # Apply selected distortion
    if mode == "soft":
        wet = np.tanh(amount * x)
    elif mode == "hard":
        threshold = np.clip(amount, 0.01, 1.0)
        wet = np.clip(x, -threshold, threshold) * (1/threshold)
    elif mode == "sine":
        wet = np.sin(x * np.pi * amount) * 0.8
    elif mode == "bitcrush":
        bits = np.round(x * (2**np.clip(amount, 1, 16))) / (2**np.clip(amount, 1, 16))
        wet = bits * 0.8  # Volume compensation
    else:
        wet = x
    
    # Mix and prevent clipping
    return np.clip((1 - mix) * dry + mix * wet, -0.99, 0.99)

