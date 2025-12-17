from mel_features import wav_to_mel
from pathlib import Path

test_clip = Path(
    r"C:\Users\withk\OneDrive\Documents\Coding\snore_detector\data\clips\snore\snore_001_0001.wav"
)

mel = wav_to_mel(test_clip)

print("Mel shape:", mel.shape)
