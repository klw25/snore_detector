import os
from pathlib import Path

import torch
import soundfile as sf
from scipy.signal import resample_poly

from config import SAMPLE_RATE, CLIP_SAMPLES

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/clips")


def load_audio(path: Path):
    audio, sr = sf.read(path, always_2d=True)

    # Convert to mono
    audio = audio.mean(axis=1)

    # Resample if needed
    if sr != SAMPLE_RATE:
        audio = resample_poly(audio, SAMPLE_RATE, sr)

    waveform = torch.from_numpy(audio).float().unsqueeze(0)
    return waveform


def save_wav(path: Path, waveform: torch.Tensor):
    audio = waveform.squeeze(0).numpy()
    sf.write(path, audio, SAMPLE_RATE)


def slice_file(wav_path: Path, output_dir: Path):
    waveform = load_audio(wav_path)

    total_samples = waveform.shape[1]
    clip_count = 0

    for start in range(0, total_samples - CLIP_SAMPLES + 1, CLIP_SAMPLES):
        clip = waveform[:, start : start + CLIP_SAMPLES]

        clip_name = f"{wav_path.stem}_{clip_count:04d}.wav"
        save_wav(output_dir / clip_name, clip)

        clip_count += 1

    return clip_count


def process_split(label: str):
    input_dir = RAW_DIR / label
    output_dir = OUTPUT_DIR / label
    output_dir.mkdir(parents=True, exist_ok=True)

    total_clips = 0

    for wav_file in input_dir.glob("*.wav"):
        count = slice_file(wav_file, output_dir)
        print(f"{wav_file.name}: {count} clips")
        total_clips += count

    print(f"Total {label} clips: {total_clips}\n")


if __name__ == "__main__":
    process_split("snore")
    process_split("non_snore")
