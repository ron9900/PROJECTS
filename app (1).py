
import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os

# Load the trained model
MODEL_PATH = "models/densenet_model.h5"
if not os.path.exists(MODEL_PATH):
    st.error("⚠️ Model file not found! Please upload 'densenet_model.h5'.")
    st.stop()

model = load_model(MODEL_PATH)

# Define image properties
IMG_SIZE = (224, 224)

# Class Labels
CLASS_LABELS = [
    "8_celled_grade_a", "8_celled_grade_b", "8_celled_grade_c",
    "morula_grade_a", "morula_grade_b", "morula_grade_c",
    "blastocyst_grade_a", "blastocyst_grade_b", "blastocyst_grade_c"
]

# Streamlit UI
st.set_page_config(page_title="Embryo Classifier", layout="centered")
st.title("🔬 Embryo Classification App")
st.write("Upload an embryo image. The model will classify its developmental stage and quality grade.\nIf not an embryo, it will return **'Not a Valid Image'**.")

# Upload image
uploaded_file = st.file_uploader("📷 Choose an image...", type=["jpg", "jpeg", "png"])

def preprocess_image(img_path):
    """Preprocess image for prediction."""
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(img_path):
    """Predict class or detect OOD (non-embryo) images."""
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)

    confidence_threshold = 0.6
    max_prob = np.max(prediction)

    if max_prob < confidence_threshold:
        return "Not a Valid Image"

    predicted_class = np.argmax(prediction)
    return CLASS_LABELS[predicted_class]

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Save temp file
    temp_path = "temp.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Prediction
    st.write("🧐 **Processing Image...**")
    prediction_result = predict_image(temp_path)

    # Display Result
    st.success(f"✅ Predicted Class: **{prediction_result}**")
