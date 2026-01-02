import torch
import coremltools as ct
from model import SnoreCNN

# ---------- PATHS ----------
PYTORCH_MODEL_PATH = "snore_cnn.pth"
COREML_MODEL_PATH = "snore_cnn.mlpackage"

# ---------- LOAD MODEL ----------
device = torch.device("cpu")
model = SnoreCNN()
model.load_state_dict(torch.load(PYTORCH_MODEL_PATH, map_location=device))
model.eval()

# ---------- EXAMPLE INPUT ----------
example_input = torch.randn(1, 1, 64, 63)  # adjust to your model input shape

# ---------- CONVERT TO TORCHSCRIPT ----------
# Trace the model
traced_model = torch.jit.trace(model, example_input)

# ---------- CONVERT TO CORE ML ----------
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.TensorType(name="input_audio", shape=example_input.shape)],
    source="pytorch",
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.iOS15
)

# ---------- SAVE ----------
mlmodel.save(COREML_MODEL_PATH)
print(f"Core ML model saved to {COREML_MODEL_PATH}")
