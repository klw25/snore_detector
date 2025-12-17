from pathlib import Path
import soundfile as sf
from config import SAMPLE_RATE, CLIP_DURATION

CLIPS_DIR = Path("data/clips")

def check_split(label):
    files = sorted((CLIPS_DIR / label).glob("*.wav"))

    print(f"\nChecking '{label}' clips")
    print(f"Total clips: {len(files)}")

    if not files:
        print("❌ No clips found")
        return

    audio, sr = sf.read(files[0], always_2d=True)
    duration = audio.shape[0] / sr
    expected_duration = CLIP_DURATION

    print(f"Sample rate: {sr} Hz")
    print(f"Duration: {duration:.2f} sec (expected {expected_duration:.2f} sec)")

    if sr != SAMPLE_RATE:
        print("⚠️ Sample rate mismatch")
    if abs(duration - expected_duration) > 0.05:
        print("⚠️ Clip duration mismatch")

if __name__ == "__main__":
    check_split("snore")
    check_split("non_snore")
