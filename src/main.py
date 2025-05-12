"""
Final Project: Generative AI Audio FX chain 
Kyle Six, KSL9443
Kahlia Gronthos, KAG9710
5/8/2025

A GUI application for generating novel audio samples from state-of-the-art generative AI, and an FX chain to tweak the output.
"""
################# Outside Modules
# TKinter
import tkinter as Tk
from tkinter import scrolledtext, filedialog, messagebox, font
from tkdial import ImageKnob
# Audio processing
import threading, time, wave
import pyaudio, torch, torchaudio
import numpy as np
# Real-time Plot
import matplotlib.figure
from matplotlib import animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
################# Project Code
# GenAi Modules
from model_interface import ModelInterface
from generate_audio import TangoFluxModel, AudioldmModel
# DSP Effects
from reverb import init_reverb, apply_reverb
from delay import init_delay, apply_delay
from distortion import apply_distortion


#################################################################
### TKinter App
#################################################################
class App:
    SAMPLE_RATE = 44100
    STREAM_DURATION = 0.1       #Note that longer this is, the more delay for UI    
    BLOCK_SIZE = int(SAMPLE_RATE * STREAM_DURATION) 
    
    def __init__(self, root, model: ModelInterface, p: pyaudio.PyAudio):
        self.root = root
        self.model = model
        self.p = p
        
        self.audio_tensor = np.zeros((2,self.BLOCK_SIZE))
        self.audio_npy = np.zeros((2,self.BLOCK_SIZE))
        self.processed_np = np.zeros((2,self.BLOCK_SIZE))
        
        self.playback_pos = 0
        
        self.audio_generated = False
        self.audio_playing = False
        
        self.dial_updated = False
        
        # Initialize GUI components
        self.setup_ui()
        self.create_parameter_controls()
    
    ###########
    ### GUI Setup
    ##########
    def set_dial_flag(self):
        if self.dial_updated == False:
            self.dial_updated = True
        
    def setup_ui(self):
        """Set up the main GUI components"""
        self.root.title("AI Audio Generator")
        self.root.geometry("500x620")
        
        self.title_label = Tk.Label(
            self.root,
            text="Generative Sampler",
            font=("Segoe UI", 18, "bold")
            )
        self.title_label.pack(side=Tk.TOP, pady=(20,10))
        
        self.knob_frame = Tk.Frame(root)
        
        self.reverb_param = Tk.DoubleVar(value=0.0)
        self.reverb_dial = ImageKnob(
            self.knob_frame, 
            image="assets/knob.png", 
            text="Reverb: ",
            progress=False,
            progress_color="cyan",
            scroll_steps=0.05,
            radius=125,
            start=1.0,
            end=0.0,
            start_angle=135,          # Pixel size of the knob
            end_angle=270,
            command=self.set_dial_flag)
        self.reverb_dial.pack(side=Tk.LEFT, padx=20)
        self.reverb_dial.set(0)
        
        self.delay_param = Tk.DoubleVar(value=0.0)
        self.delay_dial = ImageKnob(
            self.knob_frame, 
            image="assets/knob.png", 
            text="Delay: ",
            progress=False,
            progress_color="cyan",
            radius=125,
            scroll_steps=0.05,
            start=1.0,
            end=0.0,
            start_angle=135,          
            end_angle=270,
            command=self.set_dial_flag)
        self.delay_dial.pack(side=Tk.LEFT, padx=20)
        self.delay_dial.set(0)
        
        self.distortion_param = Tk.DoubleVar(value=0.0)
        self.distortion_dial = ImageKnob(
            self.knob_frame, 
            image="assets/knob.png", 
            text="Distortion: ",
            progress=False,
            progress_color="cyan",
            radius=125,
            scroll_steps=1,
            start=30,
            end=0,
            start_angle=135,
            end_angle=270,
            command=self.set_dial_flag)
        self.distortion_dial.pack(side=Tk.LEFT, padx=20)
        self.distortion_dial.set(0)
        self.knob_frame.pack(side=Tk.TOP)
        
        fig_frame = Tk.LabelFrame(self.root, text="Waveform")
        self.realtime_fig = matplotlib.figure.Figure()
        my_canvas = FigureCanvasTkAgg(self.realtime_fig, master = fig_frame)
        self.realtime_canvas = my_canvas.get_tk_widget()    # canvas widget
        self.realtime_canvas.config(width=400, height=200)    # in pixels, set canvas size to something more manageable
        self.realtime_canvas.pack(side=Tk.TOP)              # place canvas widget
        
        self.realtime_fig.patch.set_facecolor((240 / 255.0, 240/ 255.0, 237/ 255.0)) # match Tkinter bg color :)
        my_ax = self.realtime_fig.add_subplot(1, 1, 1)
        self.g1 = my_ax.plot([], [])[0]
        my_ax.set_ylim(-1.0, 1.0) # Axis limits are the min/max 32 FLOAT PCM!!!
        my_ax.set_xlim(0, self.BLOCK_SIZE)  # xlim is the number of frames per processing block
        my_ax.set_xlabel('Time (index)')
        #my_ax.set_title('Signal')
        
        # Define animation using figure, update and init functions, at the specified update interval
        self.my_anima = animation.FuncAnimation(
            self.realtime_fig,
            self.graph_update,
            init_func = self.graph_init,
            interval = 20,   # milliseconds
            blit = True,
            cache_frame_data = False,
            repeat = False
        )
        
        # Status Bar
        self.status_label = Tk.Label(
            fig_frame, 
            text="Ready", 
            bd=1, 
            relief=Tk.SUNKEN, 
            anchor=Tk.W
        )
        self.status_label.pack(fill=Tk.X, padx=5, pady=5)
        fig_frame.pack(side=Tk.TOP, padx=10, pady=10)
        
        playback_frame = Tk.Frame(self.root)
        playback_frame.pack(padx=10, pady=5)
        self.play_button = Tk.Button(
            playback_frame,
            text="Play",
            command=self.on_play
        )
        self.play_button.pack(side=Tk.LEFT)
        
        # Prompt Entry
        entry_frame = Tk.Frame(self.root)
        entry_frame.pack(padx=10, pady=5, fill=Tk.X)
        
        self.prompt_var = Tk.StringVar()
        self.entry_box = Tk.Entry(
            entry_frame, 
            textvariable=self.prompt_var, 
            font=("Helvetica", 10)
        )
        self.entry_box.pack(side=Tk.LEFT, fill=Tk.X, expand=True, padx=(0, 5))
        
        self.send_button = Tk.Button(
            entry_frame, 
            text="Generate", 
            command=self.on_generate
        )
        self.send_button.pack(side=Tk.RIGHT)
        
        # Bind Enter key to generate
        self.entry_box.bind("<Return>", lambda event: self.on_generate())
        
        # Save Button (initially disabled)
        self.save_btn = Tk.Button(
            self.root, 
            text="Save Audio", 
            state=Tk.DISABLED,
            command=self.save_audio
        )
        self.save_btn.pack(side=Tk.BOTTOM, fill=Tk.X, padx=10, pady=5)
        
    def create_parameter_controls(self):
        """Add controls for TangoFlux parameters"""
        param_frame = Tk.LabelFrame(self.root, text="Generation Parameters")
        param_frame.pack(padx=10, pady=5, fill=Tk.X)
        
        # Steps control
        Tk.Label(param_frame, text="Steps (quality):").grid(row=0, column=0, sticky="w")
        self.steps_var = Tk.IntVar(value=10)
        Tk.Spinbox(
            param_frame, 
            from_=10, 
            to=100, 
            textvariable=self.steps_var,
            width=5
        ).grid(row=0, column=1, sticky="w", padx=5)
        
        # Duration control
        Tk.Label(param_frame, text="Duration (sec):").grid(row=0, column=2, sticky="w", padx=(20,0))
        self.duration_var = Tk.DoubleVar(value=5.0)
        Tk.Spinbox(
            param_frame, 
            from_=1.0, 
            to=30.0, 
            increment=0.5,
            textvariable=self.duration_var,
            width=5
        ).grid(row=0, column=3, sticky="w", padx=5)
    
    # Setup x axis for the appropriate block size
    def graph_init(self):
        self.g1.set_xdata(range(self.BLOCK_SIZE))
        return (self.g1,)
    
    # Update function: sets the y axis data to the current output block from the main loop
    def graph_update(self, i):
        audio_np = np.mean(self.processed_np, axis=0)
        start = self.playback_pos * self.BLOCK_SIZE
        audio_block = audio_np[start : start + self.BLOCK_SIZE]
        if audio_block.shape[0] == self.BLOCK_SIZE:
            self.g1.set_ydata(audio_block)
        return (self.g1,)

    ###########
    ### Sample Generation
    ##########
    def on_generate(self):
        """Handle the generate button click"""
        prompt = self.prompt_var.get().strip()
        if not prompt:
            messagebox.showwarning("Input Error", "Please enter a text prompt")
            return
            
        self.prompt_var.set("")  # Clear the input
        #self.update_chat(prompt, "You")
        self.status_label.config(text="Generating audio...")
        self.send_button.config(state=Tk.DISABLED)
        self.audio_generated = False

        duration = self.duration_var.get()
        steps = self.steps_var.get()
        def _generate(prompt, duration, steps):
            ''' Gen AI Entrypoint '''
            try:
                # Generate audio
                audio = self.model.infer(
                    prompt,
                    duration=duration,
                    steps=steps
                )
                self.audio_tensor = audio
                self.audio_npy = audio.numpy()
                # Update UI on main thread
                self.root.after(0, lambda: self.on_generation_success(prompt))
            except Exception as e:
                self.root.after(0, lambda: self.on_generation_error(str(e)))
                
        # Run generation in background thread
        threading.Thread(target=_generate(prompt, duration, steps), daemon=True).start()
        
    def on_generation_success(self, prompt):
        """Handle successful audio generation"""
        #self.update_chat(f"Generated audio for: '{prompt}'")
        self.status_label.config(text="Playing audio...")
        self.playback_pos = 0  # Start from the beginning
        #self.setup_reverb_effect()               
        #self.play_audio_async()
        # Enable save button
        self.save_btn.config(state=Tk.NORMAL)
        #self.current_audio = audio  # Store for saving
        self.send_button.config(state=Tk.NORMAL)
        self.audio_effect_chain()
        self.audio_generated = True
        self.num_samples = self.audio_npy.shape[1] // self.BLOCK_SIZE
        self.on_play()
        
    def audio_effect_chain(self):
        board = Pedalboard([
            Reverb(room_size=0.5, damping=0.4, wet_level=self.reverb_dial.get(), dry_level=(1.0 - self.reverb_dial.get())),
            Distortion(drive_db=self.distortion_dial.get()),
            Delay(delay_seconds=0.2, feedback=self.delay_dial.get(), mix=0.3),
        ])
        self.processed_np = board(self.audio_npy, self.SAMPLE_RATE)                

    def on_play(self):
        if self.audio_playing == False:
            self.audio_playing = True
            self.play_button.config(text="Stop")
            self.status_label.config(text="Playing audio...")
        else:
            self.audio_playing = False
            self.play_button.config(text="Play")
            self.status_label.config(text="Audio stopped...")        
        
    def on_generation_error(self, error):
        """Handle generation errors"""
        #self.update_chat(f"Error: {error}")
        self.status_label.config(text=f"Error: {error}")
        self.send_button.config(state=Tk.NORMAL)
        messagebox.showerror("Generation Error", error)
    
    ###########
    ### Sample saving
    ##########
    def save_audio(self):
        """Save the generated audio to file"""
        if not self.audio_generated:
            return
            
        filepath = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
            title="Save Audio File"
        )
        
        if filepath:
            try:
                processed_tensor = torch.from_numpy(self.processed_np.T)
                torchaudio.save(filepath, self.audio_tensor, sample_rate=self.SAMPLE_RATE, )
                self.status_label.config(text=f"Saved to {filepath}")
                #self.update_chat(f"Audio saved to {filepath}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save: {str(e)}")
                pass
from pedalboard import Pedalboard, Reverb, Distortion, Delay




#################################################################
### MAIN
#################################################################
import warnings
if __name__ == "__main__":
    # Ignore many warning msgs from PyTorch & others
    warnings.filterwarnings('ignore')
    
    ### Create Tkinter root
    root = Tk.Tk()
    ### Create Audio I/O
    p = pyaudio.PyAudio()
    ### Create TangoFlux instance
    model = TangoFluxModel()
    model.load() # Load model (default weights)
    
    ### Initialize gui with App class
    app = App(root, model, p)
    ### Initialize audio stream
    stream = app.p.open(
        format=pyaudio.paFloat32,
        channels=2,
        rate=44100,
        input=False,
        output=True,
        frames_per_buffer=app.BLOCK_SIZE
    )
    
    ###########
    ### UI and Playback loop
    ##########
    CONTINUE = True
    last_frame_time = int(round(time.time() * 1000))
    while CONTINUE:
        root.update()
        current_time = int(round(time.time() * 1000)) 
        
        if app.audio_generated and app.audio_playing:
            
            if app.dial_updated:
                app.audio_effect_chain()
                app.dial_updated = False
            
            audio_np = app.processed_np
            start = app.playback_pos * app.BLOCK_SIZE
            audio_block = audio_np[:, start : start + app.BLOCK_SIZE]
            stream.write(audio_block.tobytes())
            app.playback_pos = (app.playback_pos + 1) if app.playback_pos < app.num_samples else 0
            last_frame_time = int(round(time.time() * 1000))
    
    print("Cleaning up resources...")
    stream.stop_stream()
    stream.close()
    p.close()
    p.terminate()
    print("######### Application closed ###########")