from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from src.prediction import predict

MAX_BYTES = 5 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png"}

app = FastAPI(
    title ="Car Damage Detection API",
    description=(
        "Classifies a photo of a car, taken from a front or rear corner, into one of six classes: "
        "front or rear, and normal, breakage or crushed. ResNet50, fine-tuned."
    ),
    version="1.0.0",
)


class DamageOutput(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict[str, float]

@app.get("/ping")
def ping():
    return "Hello"

@app.post("/predict", response_model=DamageOutput)
async def predict_damage(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Upload a JPEG or PNG image")

    image_bytes = await file.read(MAX_BYTES + 1)
    if len(image_bytes) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than 5 MB")

    try:
        return predict(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))