import torch
from demucs.pretrained import get_model
from demucs.audio import AudioFile
from demucs.apply import apply_model
import demucs
import os
import torchaudio
import tempfile
import io


class Demucs_VocalSeparator:

    def __init__(self):
        self.model = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sample_audio": ("AUDIO",),
                "model_name": (["htdemucs", "htdemucs_ft"], {"default": "htdemucs"})
            }
        }

    RETURN_TYPES = (
        "AUDIO",  # vocal_audio
        "AUDIO"  # instrumental_audio
    )
    RETURN_NAMES = (
        "vocal_audio",
        "instrumental_audio"
    )

    FUNCTION = "separate"
    CATEGORY = "audio"

    def separate(self, sample_audio, model_name):
        audio_tensor, sample_rate = sample_audio["waveform"], sample_audio["sample_rate"]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.model is None:
            self.model = demucs.pretrained.get_model(model_name).to(device)
            self.model.eval()

        audio_tensor = audio_tensor.to(device)
        sources = apply_model(
            self.model,
            audio_tensor,
            split=True,
            overlap=0.25,
            progress=True,
            device=device,
        )[0]
        return (
            {"waveform": sources[3].unsqueeze(0), "sample_rate": sample_rate},
            {"waveform": sources[0].unsqueeze(0), "sample_rate": sample_rate}
        )


NODE_CLASS_MAPPINGS = {
    "Demucs_VocalSeparator": Demucs_VocalSeparator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Demucs_VocalSeparator": "Demucs人声分离"
}
