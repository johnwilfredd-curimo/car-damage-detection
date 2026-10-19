import os
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from torch import nn
from torchvision import models, transforms

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
WEIGHTS = ARTIFACTS / "saved_model.pth"
WEIGHTS_URL = os.environ.get(
    "MODEL_URL",
    "https://github.com/johnwilfredd-curimo/car-damage-detection/releases/download/v1.0.0/saved_model.pth",
)

# Alphabetical order of the training folders: F_Breakage, F_Crushed, F_Normal, R_Breakage, R_Crushed, R_Normal
CLASS_NAMES = [
    "Front Breakage",
    "Front Crushed",
    "Front Normal",
    "Rear Breakage",
    "Rear Crushed",
    "Rear Normal",
]

# Same as the evaluation pipeline in the notebook: no augmentation at inference
_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_model = None

class CarClassifierResNet(nn.Module):
    def __init__(self, num_classes=6, dropout_rate=0.2):
        super().__init__()
        # weights=None: the architecture only. Our own weights are loaded straight after.
        self.model = models.resnet50(weights=None)
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.model.fc.in_features, num_classes),
        )

    def forward(self, x):
        x = self.model(x)
        return x

def ensure_weights():
    """Download the trained weights once. They are too large to keep in git."""
    if not WEIGHTS.exists():
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(WEIGHTS_URL, str(WEIGHTS))
    return WEIGHTS

def load_model():
    global _model
    if _model is None:
        model = CarClassifierResNet(num_classes=len(CLASS_NAMES))
        state = torch.load(ensure_weights(), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        _model = model
    return _model

def predict(image_bytes):
    """Classify one image given as raw bytes. Raises ValueError if the bytes are not an image."""
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("File is not a readable image") from exc

    batch = _transform(image).unsqueeze(0)
    with torch.no_grad():
        probabilities = torch.softmax(load_model()(batch), dim=1)[0]

    top = int(probabilities.argmax())
    return {
        "prediction": CLASS_NAMES[top],
        "confidence": round(float(probabilities[top]), 4),
        "probabilities": {
            name: round(float(p), 4) for name, p in zip(CLASS_NAMES, probabilities)
        },
    }