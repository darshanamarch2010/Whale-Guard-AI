import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
IMG_SIZE = 128
MODEL_PATH = "whale_detector.keras"
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)
def predict(model, image: Image.Image):
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
    arr = np.array(img, dtype="float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    score = float(model.predict(arr, verbose=0)[0][0])
    label = "🐋 Whale detected" if score > 0.5 else "🚢 No whale (ship/other)"
    confidence = score if score > 0.5 else 1 - score
    return label, confidence
st.set_page_config(page_title="Whale Detector", page_icon="🐋")
st.title("🐋 Whale Detector")
st.write(
    "Upload a satellite image chip and this model will predict whether "
    "it contains a whale or not."
)
model = load_model()
uploaded_file = st.file_uploader(
    "Upload a satellite image", type=["jpg", "jpeg", "png"]
)
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_column_width=True)
 with st.spinner("Predicting...")
        label, confidence = predict(model, image)
    st.subheader(label)
    st.write(f"Confidence: {confidence:.1%}")
else:
    st.info("Upload an image above to get a prediction.")
