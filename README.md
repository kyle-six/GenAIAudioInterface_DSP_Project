#🎶 Generative Audio Sampler
A real-time desktop application that uses generative AI models to synthesize audio from text prompts, with an interactive FX chain (reverb, delay, distortion) and waveform visualization. Built using Python, PyTorch, and Tkinter, this tool offers a hands-on way to explore creative sound synthesis.

<video src="https://github.com/kyle-six/GenAIAudioInterface_DSP_Project/blob/06bb244de849d2c55161158427693cfaf61c0a6d/assets/ProjectDemoVideo_GenAiSampler.mp4"></video>

#Description
This project combines state-of-the-art generative audio models (e.g., TangoFlux) with an interactive GUI. Users can enter a text prompt to generate unique audio clips and apply real-time audio effects using rotary knobs. The app supports playback, waveform visualization, and saving of generated audio.

#Installation
Prerequisites:
    Python 3.8+
    Anaconda (Or venv is okay)
    CUDA 12.4 (If you want GPU acceleration, otherwise will use CPU)
    Tkinter (usually included with Python)
    Pytorch v2.4.0 compatibiltiy (NO intel-based MACs!- Check your compatibility on the Pytorch website)

Instructions:
```PowerShell
git clone https://github.com/kyle-six/GenAIAudioInterface_DSP_Project.git
```
Clone our repo to wherever you like
```PowerShell
cd {repo_location}
```
Change directory to this location
```PowerShell
conda create gensampler_env python=3.9
conda activate gensampler_env
```
Create and activate a virtual environment to isolate the big GenAI dependencies for easy cleanup
```PowerShell
python3 -m pip install -r requirements.txt
```
Install the project requirements. (Note: this will take a while)

#Running the Application
```PowerShell
python3 src/main.py
```
On launch, the GUI window will open. You can:
- Enter a text prompt to generate a sound clip. (You must wait while the inference runs!)
- Click Play/Stop to hear the audio.
- Scroll over the interactive knobs to adjust reverb, delay, and distortion.
- View the real-time waveform.
- Click Save to export it as a .wav file.
