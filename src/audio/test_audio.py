import torch
import torchaudio

from config import SAMPLE_RATE, CLIP_SAMPLES

# Generate 1 second of fake audio
waveform = torch.randn(1, CLIP_SAMPLES)

print("Waveform shape:", waveform.shape)
print("Sample rate:", SAMPLE_RATE)

# Create a Mel Spectrogram transform
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024,
    hop_length=256,
    n_mels=64,
)

mel_spec = mel_transform(waveform)

print("Mel spectrogram shape:", mel_spec.shape)
