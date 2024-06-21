
import torch
import librosa
import numpy as np
# import ipywidgets as widgets
import openvino as ov
from pathlib import Path
import sounddevice as sd
import scipy.io.wavfile as wav
from transformers import Wav2Vec2ForCTC, AutoProcessor

model_id = "facebook/mms-1b-all"
MAX_SEQ_LENGTH = 30480
core = ov.Core()

# device = widgets.Dropdown(
#     options=core.available_devices + ["AUTO"],
#     value="AUTO",
#     description="Device:",
#     disabled=False,
# )
device = "CPU"

asr_processor = AutoProcessor.from_pretrained(model_id)
asr_model = Wav2Vec2ForCTC.from_pretrained(model_id)

# asr_processor.tokenizer.vocab.keys()
language_id = "mal"
asr_processor.tokenizer.set_target_lang(language_id)
asr_model.load_adapter(language_id)

asr_model_xml_path_template = "models/mlm_speech2text_model.xml"


def get_asr_model(model_path_template, language_id, compiled=True):
    input_values = torch.zeros([1, MAX_SEQ_LENGTH], dtype=torch.float)
    model_path = Path(model_path_template.format(language_id))

    asr_processor.tokenizer.set_target_lang(language_id)
    if not model_path.exists() and model_path_template == asr_model_xml_path_template:
        asr_model.load_adapter(language_id)

        model_path.parent.mkdir(parents=True, exist_ok=True)
        converted_model = ov.convert_model(asr_model, example_input={"input_values": input_values})
        ov.save_model(converted_model, model_path)
        if not compiled:
            return converted_model

    if compiled:
        return core.compile_model(model_path, device_name=device)
    return core.read_model(model_path)


compiled_asr_model = get_asr_model(asr_model_xml_path_template, language_id)

