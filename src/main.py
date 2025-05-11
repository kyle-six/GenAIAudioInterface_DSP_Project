"""
Final Project: Generative AI Audio FX chain 
Kyle Six, KSL9443
Kahlia Gronthos, KAG9710
5/8/2025

A GUI application for generating novel audio samples from state-of-the-art generative AI, and an FX chain to tweak the output.
"""
import tkinter as Tk
from tkinter import scrolledtext, filedialog, messagebox
from tkdial import ImageKnob
import threading, time
import pyaudio, torch
import numpy as np
import wave

from model_interface import ModelInterface
from generate_audio import TangoFluxModel, AudioldmModel

import matplotlib.figure
from matplotlib import animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from reverb import init_reverb, apply_reverb

import pysndfx
from pedalboard import Pedalboard, Reverb, Distortion, Delay

class App:
    SAMPLE_RATE = 44100
    STREAM_DURATION = 0.01         #Note that longer this is, the more delay for UI    
    BLOCK_SIZE = int(SAMPLE_RATE * STREAM_DURATION) 
    
    def __init__(self, root, model: ModelInterface, p: pyaudio.PyAudio):
        self.root = root
        self.model = model
        self.p = p
        
        self.audio_generated = False
        self.audio_playing = False
        
        # Initialize GUI components
        self.setup_ui()
        self.create_parameter_controls()
        
    def setup_ui(self):
        """Set up the main GUI components"""
        self.root.title("AI Audio Generator")
        self.root.geometry("500x600")
        
        self.title_label = Tk.Label(
            self.root,
            text="GenSampler",
            font="-weight bold"
            )
        self.title_label.pack(side=Tk.TOP, pady=20)
        
        self.knob_frame = Tk.Frame(root)
        
        self.reverb_param = Tk.DoubleVar(value=0.0)
        self.reverb_dial = ImageKnob(
            self.knob_frame, 
            image="assets/knob.png", 
            text="Reverb: ",
            #scale_image="assets/fun_scale_transparent.png",
            progress=False,
            progress_color="cyan",
            scroll_steps=0.05,
            radius=125,
            start=1.0,
            end=0.0,
            start_angle=135,          # Pixel size of the knob
            end_angle=270)
        self.reverb_dial.pack(side=Tk.LEFT, padx=20)
        self.reverb_dial.set(0)
        
        self.delay_param = Tk.DoubleVar(value=0.0)
        self.delay_dial = ImageKnob(
            self.knob_frame, 
            image="assets/knob.png", 
            text="Delay: ",
            #scale_image="assets/fun_scale_transparent.png",
            progress=False,
            progress_color="cyan",
            radius=125,
            scroll_steps=0.05,
            start=1.0,
            end=0.0,
            start_angle=135,          # Pixel size of the knob
            end_angle=270)
        self.delay_dial.pack(side=Tk.LEFT, padx=20)
        self.delay_dial.set(0)
        
        self.distortion_param = Tk.DoubleVar(value=0.0)
        self.distortion_dial = ImageKnob(
            self.knob_frame, 
            image="assets/knob.png", 
            text="Distortion: ",
            #scale_image="assets/fun_scale_transparent.png",
            progress=False,
            progress_color="cyan",
            radius=125,
            scroll_steps=1,
            start=30,
            end=0,
            start_angle=135,          # Pixel size of the knob
            end_angle=270)
        self.distortion_dial.pack(side=Tk.LEFT, padx=20)
        self.distortion_dial.set(0)
        self.knob_frame.pack(side=Tk.TOP)
        
        fig_frame = Tk.LabelFrame(self.root, text="Waveform")
        self.realtime_fig = matplotlib.figure.Figure()
        my_canvas = FigureCanvasTkAgg(self.realtime_fig, master = fig_frame)
        self.realtime_canvas = my_canvas.get_tk_widget()    # canvas widget
        self.realtime_canvas.config(width=400, height=200)    # in pixels, set canvas size to something more manageable
        self.realtime_canvas.pack(side=Tk.TOP)              # place canvas widget
        
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
        
        # Prompt Entry
        entry_frame = Tk.Frame(self.root)
        entry_frame.pack(padx=10, pady=5, fill=Tk.X)
        
        self.prompt_var = Tk.StringVar()
        self.entry_box = Tk.Entry(
            entry_frame, 
            textvariable=self.prompt_var, 
            font=("Helvetica", 12)
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
        
    # def update_chat(self, message, sender="AI"):
    #     """Update the chat display with a new message"""
    #     self.chat_display.config(state="normal")
    #     self.chat_display.insert(Tk.END, f"{sender}: {message}\n")
    #     self.chat_display.config(state="disabled")
    #     self.chat_display.yview(Tk.END)
        
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
        
        # audio = self.model.infer(
        #             prompt,
        #             duration=self.duration_var.get(),
        #             steps=self.steps_var.get()
        #         )
        
        # self.on_generation_success(audio, prompt)
        duration = self.duration_var.get()
        steps = self.steps_var.get()
        
        def _generate(prompt, duration, steps):
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
        
        self.audio_generated = True
        self.num_samples = self.audio_npy.shape[0]
        self.audio_playing = True
        
        
    def on_generation_error(self, error):
        """Handle generation errors"""
        #self.update_chat(f"Error: {error}")
        self.status_label.config(text=f"Error: {error}")
        self.send_button.config(state=Tk.NORMAL)
        messagebox.showerror("Generation Error", error)
        
    def save_audio(self):
        """Save the generated audio to file"""
        if not hasattr(self, 'current_audio'):
            return
            
        filepath = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
            title="Save Audio File"
        )
        
        if filepath:
            try:
                self.generator.save_wav(self.current_audio, filepath)
                self.status_label.config(text=f"Saved to {filepath}")
                self.update_chat(f"Audio saved to {filepath}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save: {str(e)}")
    
    # def playback(self):
    #     """Play the most recent generated audio with the desired FX """
        
    #     output_block = [int(x * 32767) for x in output_block] # convert output block from float
    #     pass
    
    # def play_current_audio(self):
        
    #     pass

    def play_audio_async(self, sample_rate: int = 44100, on_complete: callable = None):
        """
        Plays a PyTorch audio tensor in the background using PyAudio and calls `on_complete` when done.

        Args:
            audio_tensor (torch.Tensor): Audio tensor of shape (channels, samples), dtype float32.
            sample_rate (int): Sampling rate of the audio.
            on_complete (callable, optional): Function to call after playback is finished.
        """
        audio_tensor = self.audio_tensor
        assert audio_tensor.ndim == 2, "Expected audio tensor with shape (channels, samples)"
        assert audio_tensor.dtype == torch.float32, "Expected float32 audio tensor"

        def _play():
            try:
                # Convert tensor to numpy int16 PCM format
                # audio_np = audio_tensor.clamp(-1.0, 1.0).numpy()
                # audio_np = (audio_np.T * 32767).astype(np.int16)  # Shape: (samples, channels)
                
                audio_np = audio_tensor.numpy()
                # Step 3: Create and apply the Pedalboard effects
                board = Pedalboard([
                    Reverb(room_size=1.0, damping=0.7, wet_level=self.reverb_dial.get(), dry_level=(1.0 - self.reverb_dial.get())),
                    Distortion(drive_db=self.distortion_dial.get()),
                    Delay(delay_seconds=0.3, feedback=self.delay_dial.get(), mix=0.8),
                ])

                processed_np = board(audio_np, sample_rate)                
                audio_bytes = processed_np.tobytes()

                # Set up PyAudio stream
                p = pyaudio.PyAudio()
                stream = p.open(
                    format=pyaudio.paFloat32,
                    channels=audio_tensor.shape[0],
                    rate=sample_rate,
                    output=True
                )

                stream.write(audio_bytes)
                stream.stop_stream()
                stream.close()
                p.terminate()

            finally:
                if callable(on_complete):
                    on_complete()

        thread = threading.Thread(target=_play, daemon=True)
        thread.start()


import warnings
if __name__ == "__main__":
    # Ignore many warning msgs from PyTorch & others
    warnings.filterwarnings('ignore')
    
    root = Tk.Tk()
    model = TangoFluxModel()
    p = pyaudio.PyAudio()
    model.load() # Load default model
    app = App(root, model, p)
    
    stream = app.p.open(
        format=pyaudio.paFloat32,
        channels=2,
        rate=44100,
        output=True,
        frames_per_buffer=app.BLOCK_SIZE
    )
    
    while True:
        root.update()
        
        if app.audio_generated and app.audio_playing:
            print(time.strftime("%M:%S"))
            
            audio_np = app.audio_npy
            audio_block = audio_np[app.playback_pos : app.playback_pos + app.BLOCK_SIZE]
            
            # Step 3: Create and apply the Pedalboard effects
            board = Pedalboard([
                Reverb(room_size=1.0, damping=0.7, wet_level=app.reverb_dial.get(), dry_level=(1.0 - app.reverb_dial.get())),
                Distortion(drive_db=app.distortion_dial.get()),
                Delay(delay_seconds=0.3, feedback=app.delay_dial.get(), mix=0.8),
            ])
            processed_np = board(audio_block, app.SAMPLE_RATE)                
            audio_bytes = processed_np.tobytes()
            
            stream.write(audio_bytes)
            app.playback_pos = (app.playback_pos + 1) % (app.num_samples)
            print(time.strftime("%M:%S"))
        else:
            stream.write(np.zeros((2,app.BLOCK_SIZE)).tobytes())

        
        