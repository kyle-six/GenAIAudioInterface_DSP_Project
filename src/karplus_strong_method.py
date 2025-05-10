import random

'''
Generates N random floats around [-1, 1] 

Used as excitation noise for the Karplus-Strong method
'''
def generate_input_noise(N: int = 200):
    input = [random.uniform(-1, 1) for _ in range(N)]
    return input

''' 
Uses the Karplus-Strong Method to generate an output signal of specified length, from an input buffer of random noise

Note: Modifies the input buffer to allow for sequential calls to retrieve the next block of output signal
'''
def karplus_strong_alg(noise_buffer: list, block_length: int, K: float = 0.990):
    # Initiliaze an output buffer
    output_signal = [0] * block_length
    # Iterate over the remaining noise, to fill output signal by apply K-Strong alg
    for i in range(block_length):
        # Grab the first 2 noise values to perform sequential decay 
        first = noise_buffer[0]
        second = noise_buffer[1] if len(noise_buffer) > 1 else 0
        # Scaled by K value, get average between 1st and 2nd noise values
        avg = K * 0.5 * (first + second)
        output_signal[i] = first # take the first as the output
        noise_buffer.append(avg) # place new avg at end of noise buffer
        noise_buffer.pop(0) # remove old value
    return output_signal