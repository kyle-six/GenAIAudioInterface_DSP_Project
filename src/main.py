'''
Final Project: Generative AI Audio FX chain 
Kyle Six, KSL9443
Kahlia Gronthos, KAG9710
5/8/2025

[Description]

Useage Instructions:


Citations:

'''

"""
Final Project: Generative AI Audio FX chain 
Kyle Six, KSL9443
Kahlia Gronthos, KAG9710
5/8/2025

A GUI application for generating audio from text prompts using TangoFlux.
"""

import tkinter as Tk
from tkinter import scrolledtext, filedialog, messagebox
import threading
from generate_audio import TangoFluxGenerator

class AudioApp:
    def __init__(self, root):
        self.root = root
        self.generator = TangoFluxGenerator()
        
        # Initialize GUI components
        self.setup_ui()
        self.create_parameter_controls()
        
    def setup_ui(self):
        """Set up the main GUI components"""
        self.root.title("AI Audio Generator")
        self.root.geometry("800x600")
        
        # Chat Display
        self.chat_display = scrolledtext.ScrolledText(
            self.root, 
            wrap=Tk.WORD, 
            state="disabled", 
            font=("Helvetica", 12)
        )
        self.chat_display.pack(padx=10, pady=10, fill=Tk.BOTH, expand=True)
        
        # Status Bar
        self.status_label = Tk.Label(
            self.root, 
            text="Ready", 
            bd=1, 
            relief=Tk.SUNKEN, 
            anchor=Tk.W
        )
        self.status_label.pack(fill=Tk.X, padx=5, pady=5)
        
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
        self.steps_var = Tk.IntVar(value=50)
        Tk.Spinbox(
            param_frame, 
            from_=10, 
            to=100, 
            textvariable=self.steps_var,
            width=5
        ).grid(row=0, column=1, sticky="w", padx=5)
        
        # Duration control
        Tk.Label(param_frame, text="Duration (sec):").grid(row=0, column=2, sticky="w", padx=(20,0))
        self.duration_var = Tk.DoubleVar(value=10.0)
        Tk.Spinbox(
            param_frame, 
            from_=1.0, 
            to=30.0, 
            increment=0.5,
            textvariable=self.duration_var,
            width=5
        ).grid(row=0, column=3, sticky="w", padx=5)
        
    def update_chat(self, message, sender="AI"):
        """Update the chat display with a new message"""
        self.chat_display.config(state="normal")
        self.chat_display.insert(Tk.END, f"{sender}: {message}\n")
        self.chat_display.config(state="disabled")
        self.chat_display.yview(Tk.END)
        
    def on_generate(self):
        """Handle the generate button click"""
        prompt = self.prompt_var.get().strip()
        if not prompt:
            messagebox.showwarning("Input Error", "Please enter a text prompt")
            return
            
        self.prompt_var.set("")  # Clear the input
        self.update_chat(prompt, "You")
        self.status_label.config(text="Generating audio...")
        self.send_button.config(state=Tk.DISABLED)
        
        def _generate():
            try:
                # Generate audio
                audio = self.generator.generate(
                    prompt,
                    steps=self.steps_var.get(),
                    duration=self.duration_var.get()
                )
                
                # Update UI on main thread
                self.root.after(0, lambda: self.on_generation_success(audio, prompt))
                
            except Exception as e:
                self.root.after(0, lambda: self.on_generation_error(str(e)))
                
        # Run generation in background thread
        threading.Thread(target=_generate, daemon=True).start()
        
    def on_generation_success(self, audio, prompt):
        """Handle successful audio generation"""
        self.update_chat(f"Generated audio for: '{prompt}'")
        self.status_label.config(text="Playing audio...")
        
        # Play audio in background
        def _play():
            self.generator.play_audio(audio)
            self.root.after(0, lambda: self.status_label.config(text="Ready"))
            
        threading.Thread(target=_play, daemon=True).start()
        
        # Enable save button
        self.save_btn.config(state=Tk.NORMAL)
        self.current_audio = audio  # Store for saving
        self.send_button.config(state=Tk.NORMAL)
        
    def on_generation_error(self, error):
        """Handle generation errors"""
        self.update_chat(f"Error: {error}")
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

if __name__ == "__main__":
    root = Tk.Tk()
    app = AudioApp(root)
    root.mainloop()