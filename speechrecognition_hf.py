import torch
import librosa
import numpy as np
import sounddevice as sd
from transformers import Wav2Vec2ForCTC, AutoProcessor

def record_audio(duration, fs):
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    print("Recording complete.")
    return audio

def transcribe_audio(audio_data, fs, processor, model, language_id):
    # Convert audio to float32 and normalize
    audio_data = audio_data.astype(np.float32) / np.iinfo(np.int16).max
    
    # Process the audio
    audio_array = librosa.resample(audio_data.flatten(), orig_sr=fs, target_sr=16000)
    
    processor.tokenizer.set_target_lang(language_id)
    inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs).logits

    ids = torch.argmax(outputs, dim=-1)[0]
    transcription = processor.decode(ids)
    # transcription = processor.batch_decode(ids)
    return transcription


def main():
    # Parameters
    duration = 5  # seconds
    fs = 16000  # Sample rate
    model_id = "facebook/mms-1b-all"
    language_id = "mal"

    # Load model and processor once
    asr_processor = AutoProcessor.from_pretrained(model_id)
    asr_model = Wav2Vec2ForCTC.from_pretrained(model_id)
    asr_model.load_adapter(language_id)

    while True:
        # Record audio
        audio_data = record_audio(duration, fs)

        # Transcribe audio
        transcription = transcribe_audio(audio_data, fs, asr_processor, asr_model, language_id)
        print("Transcription:", transcription)


if __name__ == "__main__":
    main()
