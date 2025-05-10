""" Kyle Six
    Question 2) Real-Time Plotting
    4/8/2025
    
    Resources Used:
    - PyAudio: https://people.csail.mit.edu/hubert/pyaudio/docs/
    - tKinter: https://tkdocs.com/tutorial/index.html
    - Course Resources: Demo 10, Demo 12, Demo 20, Demo 56, Demo 61
"""

from math import cos, pi 
import pyaudio, struct
import tkinter as Tk
from collections import deque
from karplus_strong_method import generate_input_noise, karplus_strong_alg
from vibrato_method import vibrato_alg

import matplotlib.figure
from matplotlib import animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Define Tkinter root
root = Tk.Tk()
root.title("Real-Time Plotting")

# Define Tk variables
f1 = Tk.DoubleVar()     # f1 : frequency in Hz
# Karplus-Strong Params
K = Tk.DoubleVar()
N = Tk.IntVar()
# Vibrato Params
f0 = Tk.DoubleVar() # LFO frequency in Hz
W = Tk.DoubleVar()  # W : Sweep width (in units of seconds)

# Initialize Tk variables
f1.set(200)   
K.set(0.990)
N.set(200)
f0.set(1.0)
W.set(0.015)

# Other Globals
PLUCK = False
white_keys = ["C", "D", "E", "F", "G", "A", "B"]
black_keys = ["C#", "D#", "EMPTY", "F#", "G#", "A#", "EMPTY"]
key_to_freq = {
    "C": 261.63,
    "C#": 277.18,
    "D": 293.67,
    "D#": 311.13,
    "E": 329.63,
    "F": 349.23,
    "F#": 369.99,
    "G": 392.0,
    "G#": 415.3,
    "A": 440.0,
    "A#": 466.16,
    "B": 493.88,
}
key_width = 10

### Callback Functions ###
def fun_quit():
    global CONTINUE
    print("Exiting Plucked String Simulator... Goodbye!")
    CONTINUE = False
  
def fun_freq_pluck(value=None):
    global PLUCK
    # Set new N for the desired freq and enable new pluck
    N.set(SAMPLE_RATE//f1.get())
    PLUCK = True

def fun_N_pluck(value=None):
    global PLUCK
     # Set new f1 for the desired N value and enable new pluck
    f1.set(SAMPLE_RATE/N.get())
    PLUCK = True

def fun_W_pluck(value=None):
    global PLUCK
    PLUCK = True

def fun_keyboard(key_str: str):
    global f1, PLUCK
    f1.set(key_to_freq[key_str])
    fun_freq_pluck() # Calculate and set N
    print(f"You pressed the {key_str} key!")

# Let the User play the keyboard using their... keyboard!
def fun_input(event):
    if event.char == 'a':
        fun_keyboard("C")
    elif event.char == 's':
        fun_keyboard("D")
    elif event.char == 'd':
        fun_keyboard("E")
    elif event.char == 'f':
        fun_keyboard("F")
    elif event.char == 'g':
        fun_keyboard("G")
    elif event.char == 'h':
        fun_keyboard("A")
    elif event.char == 'j':
        fun_keyboard("B")
    elif event.char == 'w':
        fun_keyboard("C#")
    elif event.char == 'e':
        fun_keyboard("D#")
    elif event.char == 't':
        fun_keyboard("F#")
    elif event.char == 'y':
        fun_keyboard("G#")
    elif event.char == 'u':
        fun_keyboard("A#")
# Bind the keypress events to the input fn
root.bind("<Key>", fun_input)


### Define widgets ###
F_sliders = Tk.Frame(root)
S_freq = Tk.Scale(root, orient='horizontal', label = 'Frequency', variable = f1, resolution=.01, from_ = 100, to = 500, tickinterval = 100, command=fun_freq_pluck)
L_ks_method = Tk.Label(F_sliders, text= "Karplus-Strong: ", font=('TkDefaultFont', 10, 'bold'))
S_k = Tk.Scale(F_sliders, label = 'K', variable = K, from_ = 1, to = 0.95, resolution=.0005, tickinterval = .01)
S_n = Tk.Scale(F_sliders, label = 'N', variable = N, from_ = 440, to = 40, tickinterval = 100, command=fun_N_pluck)
L_vibrato= Tk.Label(F_sliders, text= "Vibrato: ", font=('TkDefaultFont', 10, 'bold'))
S_lowfreq = Tk.Scale(F_sliders, label = 'F0', variable = f0, from_ = 25, to = 0, resolution=.1, tickinterval = 5)
S_width = Tk.Scale(F_sliders, label = 'W', variable = W, from_ = .25, to = 0, resolution=.001, tickinterval = .05, command=fun_W_pluck)
B_quit = Tk.Button(root, text = 'Quit', command = fun_quit)

# Add Row of "black piano keys" each with a different note to a frame
F_black_keyboard = Tk.Frame(root)
black_buttons = []
# Add blank, half-width placeholder label to add offset for black keys
Tk.Label(F_black_keyboard, width=key_width//2, text="").pack(side = Tk.LEFT, fill=Tk.X)
for key_str in black_keys:
    if key_str == "EMPTY":
        # Add half-width placeholder
        Tk.Label(F_black_keyboard, width=key_width, text="").pack(side = Tk.LEFT, fill=Tk.X)
    else:
        b = Tk.Button(F_black_keyboard,
                    width= key_width,
                    bg="black",
                    fg="white",
                    text=key_str, 
                    command=lambda x=key_str: fun_keyboard(x))
        b.pack(side = Tk.LEFT)
        black_buttons.append(b)
# Add Row of "white piano keys" each with a different note to a frame
F_white_keyboard = Tk.Frame(root)
white_buttons = []
for key_str in white_keys:
    b = Tk.Button(F_white_keyboard,
                  width= key_width,
                  bg="white",
                  text=key_str, 
                  command=lambda x=key_str: fun_keyboard(x))
    b.pack(side = Tk.LEFT)
    white_buttons.append(b)

### Place widgets ###
########## Declare NEW plotting elements ######## will customize later
my_fig = matplotlib.figure.Figure()
my_canvas = FigureCanvasTkAgg(my_fig, master = root)
C1 = my_canvas.get_tk_widget()    # canvas widget
C1.config(width=600, height=400)    # in pixels, set canvas size to something more manageable
# Top of Window
C1.pack(side=Tk.TOP)              # place canvas widget
L_ks_method.pack(side=Tk.LEFT)
S_k.pack(side=Tk.LEFT)
S_n.pack(side=Tk.LEFT)
L_vibrato.pack(side=Tk.LEFT)
S_lowfreq.pack(side=Tk.LEFT)
S_width.pack(side=Tk.LEFT)
F_sliders.pack(pady=10)
S_freq.pack(fill=Tk.X)

# Bottom of Window
B_quit.pack(side = Tk.BOTTOM, fill=Tk.X, pady=[10,0])
F_white_keyboard.pack(side=Tk.BOTTOM)
F_black_keyboard.pack(side=Tk.BOTTOM)


### Real-Time Signal Generation ###
SAMPLE_RATE = 44100     # sampling rate (samples/second)
DURATION = 0.01         #Note that longer this is, the more delay for UI
BLOCKLEN = int(SAMPLE_RATE * DURATION) 

# Holds the remaining noise for the Karplus-strong to decay over time
noise_buffer = [0] * BLOCKLEN  # Note this is treated as a circular bufffer via append/pop
# Holds the remaining noise for the Karplus-strong to decay over time
vibrato_buffer = [0] * int(W.get() * SAMPLE_RATE + 2)   # Note this is treated as a circular bufffer
kr = 0  # read index
kw = len(vibrato_buffer) // 2 # write index

# Holds the frames for output to audio stream
output_block = [0] * BLOCKLEN

# Create Pyaudio object
p = pyaudio.PyAudio()
stream = p.open(
    format = pyaudio.paInt16,  
    channels = 1, 
    rate = SAMPLE_RATE,
    input = False, 
    output = True,
    frames_per_buffer=BLOCKLEN)

### REAL-TIME PLOTTING
# Note that figure and canvas are defined in placement section
# Setup matplotlib figure! it was declared above for UI placement reasons
my_fig.patch.set_facecolor((240 / 255.0, 240/ 255.0, 237/ 255.0)) # match Tkinter bg color :)
my_ax = my_fig.add_subplot(1, 1, 1)
[g1] = my_ax.plot([], [])
my_ax.set_ylim(-32000, 32000) # Axis limits are the min/max 16bit inta
my_ax.set_xlim(0, BLOCKLEN)  # xlim is the number of frames per processing block
my_ax.set_xlabel('Time (index)')
my_ax.set_title('Signal')

# Setup x axis for the appropriate block size
def my_init():
  g1.set_xdata(range(BLOCKLEN))
  return (g1,)

# Update function: sets the y axis data to the current output block from the main loop
def my_update(i):
  global output_block

  g1.set_ydata(output_block)
  return (g1,)

# Define animation using figure, update and init functions, at the specified update interval
my_anima = animation.FuncAnimation(
    my_fig,
    my_update,
    init_func = my_init,
    interval = 20,   # milliseconds
    blit = True,
    cache_frame_data = False,
    repeat = False
)

################### MAIN LOOP ############################################
CONTINUE = True
print("* Program Start *")
while CONTINUE:
    root.update()
    
    ### Karplus-Strong Effect
    if PLUCK:
        # Note that N is set via sliders
        #Generate new input noise with desired N value
        noise_buffer = generate_input_noise(N.get())
        
        vibrato_buffer = [0] * int(W.get() * SAMPLE_RATE + 2) 
        kr, kw = 0, len(vibrato_buffer) // 2
        PLUCK = False
    # Perform K-S decay on any remaining noise buffer
    output_block = karplus_strong_alg(noise_buffer, BLOCKLEN, K.get())
    output_block = [int(x * 32767) for x in output_block] # convert output block from float
    
    ### Vibrato Effect
    # Perform vibrato effect on signal block, using buffer
    output_block, (kr, kw) = vibrato_alg(output_block, vibrato_buffer, (kr,kw), BLOCKLEN, f0.get(), W.get(), rate=SAMPLE_RATE)
    
    binary_data = struct.pack('h' * BLOCKLEN, *output_block) # 16 bit integer format!!
    stream.write(binary_data)

print("* Program Finish *")
stream.stop_stream()
stream.close()
p.terminate()
