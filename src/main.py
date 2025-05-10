'''
Final Project: Generative AI Audio FX chain 
Kyle Six, KSL9443
Kahlia Gronthos, KAG...
5/8/2025

[Description]

Useage Instructions:


Citations:

'''


import tkinter as Tk
from tkinter import scrolledtext, filedialog
import threading
import time

# Simulated AI function (replace with actual audio generation)
def generate_audio_response(prompt: str):
    time.sleep(1)  # Simulate processing delay
    return f"[AUDIO GENERATED for: '{prompt}']"

# GUI Setup
root = Tk.Tk()
root.title("Audio Chat Interface")
root.geometry("600x400")

# Chat Display
chat_display = scrolledtext.ScrolledText(root, wrap=Tk.WORD, state="disabled", font=("Helvetica", 10))
chat_display.pack(padx=10, pady=10, fill=Tk.BOTH)

# Prompt Entry
entry_frame = Tk.Frame(root)
entry_frame.pack(padx=10, pady=5, fill=Tk.X)

prompt_var = Tk.StringVar()
entry_box = Tk.Entry(entry_frame, textvariable=prompt_var, font=("Helvetica", 10))
entry_box.pack(side=Tk.LEFT, fill=Tk.X, expand=True, padx=(0, 5))

def send_prompt():
    prompt = prompt_var.get().strip()
    if not prompt:
        return
    prompt_var.set("")
    
    # Display user prompt
    chat_display.config(state="normal")
    chat_display.insert(Tk.END, f"You: {prompt}\n")
    chat_display.config(state="disabled")
    chat_display.yview(Tk.END)
    
    # Handle AI generation in a background thread
    def process_prompt():
        response = generate_audio_response(prompt)
        chat_display.config(state="normal")
        chat_display.insert(Tk.END, f"AI: {response}\n")
        chat_display.config(state="disabled")
        chat_display.yview(Tk.END)
    
    threading.Thread(target=process_prompt, daemon=True).start()

send_button = Tk.Button(entry_frame, text="Send", command=send_prompt)
send_button.pack(side=Tk.RIGHT)

# Allow pressing Enter to send
def on_enter(event):
    send_prompt()
entry_box.bind("<Return>", on_enter)

# Quit Button
quit_button = Tk.Button(root, text="Quit", command=root.quit)
quit_button.pack(side=Tk.BOTTOM, fill=Tk.X, pady=(5, 10))

root.mainloop()
