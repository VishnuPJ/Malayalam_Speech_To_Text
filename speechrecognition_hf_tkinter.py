import torch
import librosa
import numpy as np
import sounddevice as sd
from transformers import Wav2Vec2ForCTC, AutoProcessor
import tkinter as tk
import threading

class SpeechRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Speech Recognition App")
        self.is_recording = False
        self.audio_data = []
        self.fs = 16000  # Sample rate

        # Load model and processor once
        self.model_id = "facebook/mms-1b-all"
        self.language_id = "mal"
        self.asr_processor = AutoProcessor.from_pretrained(self.model_id)
        self.asr_model = Wav2Vec2ForCTC.from_pretrained(self.model_id)
        self.asr_model.load_adapter(self.language_id)

        self.create_widgets()

    def create_widgets(self):
        self.start_button = tk.Button(self.root, text="Start", command=self.start_recording)
        self.start_button.pack(pady=10)

        self.stop_button = tk.Button(self.root, text="Stop", command=self.stop_recording, state=tk.DISABLED)
        self.stop_button.pack(pady=10)

        self.status_label = tk.Label(self.root, text="Status: Idle", fg="blue")
        self.status_label.pack(pady=10)

        self.submit_button = tk.Button(self.root, text="Submit", command=self.submit_audio, state=tk.DISABLED)
        self.submit_button.pack(pady=10)

        self.transcription_label = tk.Label(self.root, text="Transcription will appear here.", wraplength=400, justify="left")
        self.transcription_label.pack(pady=10)

    def start_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.audio_data = []
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.status_label.config(text="Status: Recording...", fg="red")
            self.recording_thread = threading.Thread(target=self.record_audio)
            self.recording_thread.start()

    def record_audio(self):
        with sd.InputStream(samplerate=self.fs, channels=1, dtype='int16', callback=self.audio_callback):
            while self.is_recording:
                sd.sleep(100)

    def audio_callback(self, indata, frames, time, status):
        self.audio_data.append(indata.copy())

    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.submit_button.config(state=tk.NORMAL)
            self.status_label.config(text="Status: Recorded, ready to submit", fg="green")

    def submit_audio(self):
        if not self.audio_data:
            self.status_label.config(text="Status: No audio recorded!", fg="red")
            return

        audio_data = np.concatenate(self.audio_data)
        transcription = self.transcribe_audio(audio_data)
        self.transcription_label.config(text=f"Transcription: {transcription}")

    def transcribe_audio(self, audio_data):
        # Convert audio to float32 and normalize
        audio_data = audio_data.astype(np.float32) / np.iinfo(np.int16).max
        
        # Process the audio
        audio_array = librosa.resample(audio_data.flatten(), orig_sr=self.fs, target_sr=16000)
        
        self.asr_processor.tokenizer.set_target_lang(self.language_id)
        inputs = self.asr_processor(audio_array, sampling_rate=16000, return_tensors="pt")

        with torch.no_grad():
            outputs = self.asr_model(**inputs).logits

        ids = torch.argmax(outputs, dim=-1)[0]
        transcription = self.asr_processor.decode(ids)
        print(transcription)
        return transcription

if __name__ == "__main__":
    root = tk.Tk()
    app = SpeechRecognitionApp(root)
    root.mainloop()
