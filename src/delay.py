# Real-time delay effect

import numpy as np

def init_delay(delay_seconds, rate, block_size):
    delay_samples = int(delay_seconds * rate)
    buffer = np.zeros(delay_samples)
    idx = 0
    return buffer, idx

def apply_delay(x_block, buffer, idx, feedback=0.4, wet_mix=0.5):
    out_block = np.zeros_like(x_block)
    buffer_len = len(buffer)
    
    for i in range(len(x_block)):
        delayed_sample = buffer[idx]
        out_block[i] = (1 - wet_mix) * x_block[i] + wet_mix * delayed_sample
        
        # Feedback + insert current sample into buffer
        buffer[idx] = x_block[i] + delayed_sample * feedback
        idx = (idx + 1) % buffer_len
    
    return out_block, buffer, idx


