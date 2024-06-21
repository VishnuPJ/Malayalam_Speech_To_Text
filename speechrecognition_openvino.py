import torch
import librosa
import numpy as np
import openvino as ov
import sounddevice as sd
from transformers import Wav2Vec2ForCTC, AutoProcessor
from pathlib import Path
device = "CPU"

core = ov.Core()

def record_audio(duration, fs):
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    print("Recording complete.")
    return audio

def get_asr_model(model_path_template, language_id, asr_processor,core,compiled=True):
    # input_values = torch.zeros([1, MAX_SEQ_LENGTH], dtype=torch.float)
    model_path = Path(model_path_template.format(language_id))
    asr_processor.tokenizer.set_target_lang(language_id)
    if compiled:
        return core.compile_model(model_path, device_name=device)
    return core.read_model(model_path)

def recognize_audio(compiled_model, src_audio,asr_processor):
    audio_data = src_audio.astype(np.float32) / np.iinfo(np.int16).max
    
    # Process the audio
    audio_array = librosa.resample(audio_data.flatten(), orig_sr=16000, target_sr=16000)
    inputs = asr_processor(audio_array, sampling_rate=16_000, return_tensors="pt")
    outputs = compiled_model(inputs["input_values"])[0]

    ids = torch.argmax(torch.from_numpy(outputs), dim=-1)[0]
    transcription = asr_processor.decode(ids)

    return transcription


def main():
    # Parameters
    duration = 5  # seconds
    fs = 16000  # Sample rate
    model_id = "facebook/mms-1b-all"
    language_id = "mal"
    asr_model_xml_path_template = "models/mlm_speech2text_model.xml"

    # Load model and processor once
    asr_processor = AutoProcessor.from_pretrained(model_id)
    # asr_model = Wav2Vec2ForCTC.from_pretrained(model_id)
    # asr_model.load_adapter(language_id)
    compiled_asr_model = get_asr_model(asr_model_xml_path_template, language_id , asr_processor , core)
    while True:
        # Record audio
        audio_data = record_audio(duration, fs)
        # Transcribe audio
        transcription = recognize_audio(compiled_asr_model, audio_data, asr_processor)
        print("Transcription:", transcription)


if __name__ == "__main__":
    main()
