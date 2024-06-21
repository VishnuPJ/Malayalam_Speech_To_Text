# Malayalam_Speech_To_Text

This Python script records audio from the microphone, processes it, and transcribes it using Facebook's Wav2Vec2 model.

I have tried two approaches:
  1) Using direct Hugging face library.
  2) Using Intel OpenVINO Intermediate Representation.

Please check the model details: [mms-1b-all](https://huggingface.co/facebook/mms-1b-all)
## Prerequisites

Make sure you have the following libraries installed:

```sh
pip install -q --upgrade pip
pip install -q "transformers>=4.33.1" "torch>=2.1" "openvino>=2023.1.0" "numpy>=1.21.0" "nncf>=2.9.0"
pip install -q --extra-index-url https://download.pytorch.org/whl/cpu torch "datasets>=2.14.6" accelerate soundfile librosa "gradio>=4.19" jiwer
```

For more details, refer to the following documentation: [OpenVINO Documentation](https://docs.openvino.ai/2024/notebooks/mms-massively-multilingual-speech-with-output.html)

---
