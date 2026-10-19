from pathlib import Path

import pandas as pd
import streamlit as st

from src.prediction import load_model, predict

SAMPLES = Path(__file__).parent / "samples"

st.set_page_config(page_title="Car Damage Detection", page_icon="🚗")
st.title("Car Damage Detection")
st.caption(
    "Upload a photo of a car taken from a front or rear corner. "
    "The model says which end it is, and whether it is normal, broken or crushed."
)

@st.cache_resource(show_spinner="Loading the model (first visit only)...")
def get_model():
    return load_model()

get_model()

sample_files = sorted(p for p in SAMPLES.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
sample_choice = None
if sample_files:
    sample_choice = st.selectbox(
        "... or try a sample", ["(none)"] + [p.name for p in sample_files]
    )

image_bytes = None
if uploaded_file:
    image_bytes = uploaded_file.getvalue()
elif sample_choice and sample_choice != "(none)":
    image_bytes = (SAMPLES / sample_choice).read_bytes()

if image_bytes:
    try:
        result = predict(image_bytes)
    except ValueError as exc:
        st.error(str(exc))
    else:
        st.image(image_bytes, width="stretch")
        st.success(f"{result['prediction']}  ·  {result['confidence']:.0%} confident")
        st.bar_chart(pd.Series(result['probabilities'], name="probability"), horizontal=True)