import torch
from torch.utils.data import Dataset
import soundfile as sf
from pathlib import Path

# -------- CONFIG --------
SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 64
EPS = 1e-9

# -------- MEL FUNCTIONS --------
def hz_to_mel(freq):
    return 2595 * torch.log10(torch.tensor(1.0) + freq / 700)

def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)

def create_mel_filterbank(n_fft, n_mels, sample_rate):
    f_min = 0
    f_max = sample_rate / 2

    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)

    mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    bins = torch.floor((n_fft + 1) * hz_points / sample_rate).long()
    fb = torch.zeros(n_mels, n_fft // 2 + 1)

    for i in range(n_mels):
        fb[i, bins[i]:bins[i + 1]] = torch.linspace(0, 1, bins[i + 1] - bins[i])
        fb[i, bins[i + 1]:bins[i + 2]] = torch.linspace(1, 0, bins[i + 2] - bins[i + 1])

    return fb

def wav_to_mel(path):
    audio, sr = sf.read(path, always_2d=True)
    audio = audio.mean(axis=1)
    waveform = torch.from_numpy(audio).float()

    spec = torch.stft(
        waveform,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        window=torch.hann_window(N_FFT),
        return_complex=True
    )

    power = spec.abs() ** 2
    fb = create_mel_filterbank(N_FFT, N_MELS, SAMPLE_RATE)
    mel = fb @ power
    mel = torch.log(mel + EPS)

    return mel

# -------- DATASET --------
class SnoreDataset(Dataset):
    def __init__(self, root="data/clips"):
        self.samples = []
        root = Path(root)

        for label, name in enumerate(["non_snore", "snore"]):
            for wav in (root / name).glob("*.wav"):
                self.samples.append((wav, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        mel = wav_to_mel(path)
        mel = mel.unsqueeze(0)  # [1, n_mels, time]
        return mel, torch.tensor(label)
