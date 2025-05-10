import math

'''
Clips audio to 16 bit integer on either side
'''
def clip16( x ):    
    # Clipping for 16 bits
    if x > 32767:
        x = 32767
    elif x < -32768:
        x = -32768
    else:
        x = x        
    return(x)

'''
Performs sinusoidal vibrato on a given signal, by using a circular buffer
'''
def vibrato_alg(input_signal:list, vibrato_buffer: list, buffer_idx: tuple, block_length: int, f0: float = 2, W: float = 0.015, rate: int = 44100,):
    Wd = W * rate   # Wd : W in units of discrete samples (use sampling rate)
    buffer_length = len(vibrato_buffer)
    
        # Buffer (delay line) indices
    kr = buffer_idx[0]  # read index
    i1 = kr
    kw = buffer_idx[1]   # write index (initialize to middle of buffer)
    
    output_signal = []
    for n in range(0, block_length):
        # Get sample from input signal
        x0 = input_signal[n]

        # Get previous and next buffer values (since kr is fractional)
        kr_prev = int(math.floor(kr))
        frac = kr - kr_prev    # 0 <= frac < 1
        kr_next = kr_prev + 1
        if kr_next == buffer_length:
            kr_next = 0

        # Compute output value using interpolation
        y0 = (1-frac) * vibrato_buffer[kr_prev] + frac * vibrato_buffer[kr_next]

        # Update buffer
        vibrato_buffer[kw] = x0

        # Increment read index
        i1 = i1 + 1
        if i1 >= buffer_length:
            # End of buffer. Circle back to front.
            i1 = i1 - buffer_length

        kr = i1 + Wd * math.sin( 2 * math.pi * f0 * n / rate )
            # Note: kr is not integer!

        # Ensure that 0 <= kr < BUFFER_LEN
        if kr >= buffer_length:
            kr = kr - buffer_length

        # Increment write index    
        kw = kw + 1
        if kw == buffer_length:
            kw = 0
            
        output_signal.append(int(clip16(y0)))    # append new to total
    return output_signal, (kr, kw)