import numpy as np
from speechbrain.pretrained import SpeakerRecognition
import torch
import torchaudio
from torchaudio.backend import soundfile_backend

model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",savedir="pretrained_models/spkrec-ecapa-voxceleb")

def extract_embedding(audio_path):
    signal = model.load_audio(audio_path)
    embedding = model.encode_batch(signal.unsqueeze(0))
    return embedding.squeeze(0).squeeze(0).detach().cpu().numpy()

def cosine_similarity(vec1, vec2):
    v1 = torch.tensor(vec1)
    v2 = torch.tensor(vec2)
    return torch.nn.functional.cosine_similarity(v1, v2, dim=0).item()



