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
        self.current_model_name = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sample_audio": ("AUDIO",),
                "model_name": (["htdemucs", "htdemucs_ft", "htdemucs_6s", "hdemucs_mmi", "mdx", "mdx_extra", "mdx_q", "mdx_extra_q"], {"default": "htdemucs"})
            }
        }

    RETURN_TYPES = (
        "AUDIO",  # drums_audio
        "AUDIO",  # bass_audio
        "AUDIO",  # other_audio
        "AUDIO"   # vocal_audio
    )
    RETURN_NAMES = (
        "vocals",
        "other",
        "drums",
        "bass"
    )

    FUNCTION = "separate"
    CATEGORY = "audio"

    def separate(self, sample_audio, model_name):
        audio_tensor, sample_rate = sample_audio["waveform"], sample_audio["sample_rate"]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.model is None or self.current_model_name != model_name:            
            # 加载新模型并记录名称
            self.current_model_name = model_name
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
            {"waveform": sources[2].unsqueeze(0), "sample_rate": sample_rate},
            {"waveform": sources[0].unsqueeze(0), "sample_rate": sample_rate},
            {"waveform": sources[1].unsqueeze(0), "sample_rate": sample_rate}
        )


NODE_CLASS_MAPPINGS = {
    "Demucs_VocalSeparator": Demucs_VocalSeparator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Demucs_VocalSeparator": "Demucs人声分离"
}
