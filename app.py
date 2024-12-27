import streamlit as st
from PIL import Image
import numpy as np
import joblib

# Load pre-trained models
rf_model = joblib.load('brain_tumor_rf_model.pkl')
pca = joblib.load('pca_model.pkl')
scaler = joblib.load('scaler_model.pkl')

st.title("Brain Tumor Classification")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")

    # Preprocess the image
    img = img.resize((64, 64))
    img_array = np.array(img).flatten().reshape(1, -1)

    # Standardize and apply PCA
    img_scaled = scaler.transform(img_array)
    img_pca = pca.transform(img_scaled)

    # Predict the class
    prediction = rf_model.predict(img_pca)
    result = "Tumor" if prediction == 1 else "Healthy"
    st.write(f"Prediction: {result}")
