import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the trained model (make sure it's in the same folder as the app)
model = load_model('model.h5')

st.title("Image Classification: Healthy vs Humor")

# File uploader widget to upload images
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image before prediction
    image = image.resize((150, 150))
    image = np.array(image) / 255.0  # Rescale pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make the prediction
    prediction = model.predict(image)
    if prediction[0] > 0.5:
        st.write("Prediction: Humor")
    else:
        st.write("Prediction: Healthy")
