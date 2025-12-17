import torch
import coremltools as ct
from model import SnoreCNN

# ---------- CONFIG ----------
DEVICE = torch.device("cpu")
MODEL_PATH = "snore_cnn.pth"
MLMODEL_PATH = "snore_cnn.mlmodel"

# ---------- LOAD MODEL ----------
model = SnoreCNN()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ---------- SAMPLE INPUT ----------
# batch=1, channels=1, n_mels=64, time=63 (matches your test mel)
example_input = torch.randn(1, 1, 64, 63)

# ---------- CONVERT TO CORE ML ----------
traced_model = torch.jit.trace(model, example_input)
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=example_input.shape)],
)

# ---------- SAVE ----------
mlmodel.save(MLMODEL_PATH)
print(f"Core ML model saved as {MLMODEL_PATH}")
