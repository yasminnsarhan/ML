# app.py
import streamlit as st
import numpy as np
import cv2
import joblib
from PIL import Image
import io

class BrainTumorApp:
    def __init__(self):
        # Load the saved model and scaler
        self.model = joblib.load('brain_tumor_model.joblib')
        self.scaler = joblib.load('scaler.joblib')
        self.img_size = (224, 224)

    def preprocess_image(self, img):
        # Convert PIL Image to numpy array
        img_array = np.array(img.convert('L'))  # Convert to grayscale
        # Resize image
        resized = cv2.resize(img_array, self.img_size)
        # Flatten and scale features
        features = resized.flatten().reshape(1, -1)
        scaled_features = self.scaler.transform(features)
        return scaled_features

    def predict(self, img):
        # Preprocess and predict
        features = self.preprocess_image(img)
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0].max()
        return prediction, probability

def main():
    st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")
    
    st.title("Brain Tumor Classification")
    st.write("Upload a brain MRI image to check for tumors")
    
    # Initialize the classifier
    classifier = BrainTumorApp()
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Add a prediction button
        if st.button('Predict'):
            with st.spinner('Analyzing image...'):
                # Make prediction
                prediction, probability = classifier.predict(image)
                
                # Display results
                st.subheader("Results:")
                result = "Tumor Detected" if prediction == 1 else "No Tumor Detected"
                
                # Color code the result
                color = "red" if prediction == 1 else "green"
                st.markdown(f"<h3 style='color: {color};'>{result}</h3>", unsafe_allow_html=True)
                
                # Display probability
                st.write(f"Confidence: {probability*100:.2f}%")
                
                # Add interpretation
                st.write("---")
                st.write("Interpretation:")
                if prediction == 1:
                    st.warning("⚠️ The model detected patterns consistent with a brain tumor. Please consult a healthcare professional for proper medical evaluation.")
                else:
                    st.success("✅ The model did not detect patterns consistent with a brain tumor. However, always consult healthcare professionals for medical advice.")

if __name__ == '__main__':
    main()