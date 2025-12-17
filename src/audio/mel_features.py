import torch
import math
import soundfile as sf
from pathlib import Path

from config import SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MELS, LOG_MEL, EPS

def hz_to_mel(freq):
    return 2595 * math.log10(1 + freq / 700)

def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)

def create_mel_filterbank(n_fft=N_FFT, n_mels=N_MELS, f_min=0.0, f_max=None, device="cpu"):
    if f_max is None:
        f_max = SAMPLE_RATE / 2

    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)

    mel_points = torch.linspace(mel_min, mel_max, n_mels + 2, device=device)
    hz_points = mel_to_hz(mel_points)

    fft_bins = torch.floor((n_fft + 1) * hz_points / SAMPLE_RATE).long()
    filterbank = torch.zeros(n_mels, n_fft // 2 + 1, device=device)

    for i in range(n_mels):
        left = fft_bins[i]
        center = fft_bins[i + 1]
        right = fft_bins[i + 2]

        if center > left:
            filterbank[i, left:center] = torch.linspace(0, 1, center - left)
        if right > center:
            filterbank[i, center:right] = torch.linspace(1, 0, right - center)

    return filterbank

def mel_spectrogram(waveform, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS):
    window = torch.hann_window(n_fft, device=waveform.device)

    stft = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        return_complex=True
    )

    magnitude = stft.abs() ** 2
    mel_fb = create_mel_filterbank(n_fft=n_fft, n_mels=n_mels, device=waveform.device)
    mel_spec = torch.matmul(mel_fb, magnitude)

    if LOG_MEL:
        mel_spec = torch.log(mel_spec + EPS)

    return mel_spec

# 🎯 Public function to use elsewhere
def wav_to_mel(path: Path):
    audio, sr = sf.read(path, always_2d=True)
    audio = audio.mean(axis=1)  # stereo → mono
    waveform = torch.from_numpy(audio).float().unsqueeze(0)
    mel = mel_spectrogram(waveform)
    return mel
