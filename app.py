try:
    import streamlit as st
    import numpy as np
    import joblib
    from PIL import Image
    import sklearn
    import pandas as pd
except ImportError as e:
    st.error(f"Failed to import required libraries. Error: {e}")
    st.error("Please check requirements.txt and make sure all libraries are installed.")
    st.stop()

class BrainTumorApp:
    def __init__(self):
        try:
            # Load the saved model and scaler
            self.model = joblib.load('brain_tumor_model.joblib')
            self.scaler = joblib.load('scaler.joblib')
            self.img_size = (224, 224)
        except Exception as e:
            st.error(f"Error loading model files: {e}")
            st.error("Please ensure model files are present in the repository")
            st.stop()

    def preprocess_image(self, img):
        try:
            # Resize image using PIL
            img = img.resize(self.img_size)
            # Convert to grayscale
            img = img.convert('L')
            # Convert to numpy array
            img_array = np.array(img)
            # Flatten and scale features
            features = img_array.flatten().reshape(1, -1)
            scaled_features = self.scaler.transform(features)
            return scaled_features
        except Exception as e:
            st.error(f"Error preprocessing image: {e}")
            return None

    def predict(self, img):
        features = self.preprocess_image(img)
        if features is not None:
            try:
                prediction = self.model.predict(features)[0]
                probability = self.model.predict_proba(features)[0].max()
                return prediction, probability
            except Exception as e:
                st.error(f"Error making prediction: {e}")
                return None, None
        return None, None

def main():
    try:
        st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")
        
        st.title("Brain Tumor Classification")
        st.write("Upload a brain MRI image to check for tumors")
        
        # Initialize the classifier
        classifier = BrainTumorApp()
        
        # File uploader
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            try:
                # Display the uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Image', use_column_width=True)
                
                # Add a prediction button
                if st.button('Predict'):
                    with st.spinner('Analyzing image...'):
                        # Make prediction
                        prediction, probability = classifier.predict(image)
                        
                        if prediction is not None and probability is not None:
                            # Display results
                            st.subheader("Results:")
                            result = "Tumor Detected" if prediction == 1 else "No Tumor Detected"
                            
                            # Color code the result
                            color = "red" if prediction == 1 else "green"
                            st.markdown(f"<h3 style='color: {color};'>{result}</h3>", 
                                      unsafe_allow_html=True)
                            
                            # Display probability
                            st.write(f"Confidence: {probability*100:.2f}%")
                            
                            # Add interpretation
                            st.write("---")
                            st.write("Interpretation:")
                            if prediction == 1:
                                st.warning("⚠️ The model detected patterns consistent with a brain tumor.")
                            else:
                                st.success("✅ The model did not detect patterns consistent with a brain tumor.")
                            
                            st.info("Please consult healthcare professionals for proper medical evaluation.")
                            
            except Exception as e:
                st.error(f"Error processing image: {e}")
                
    except Exception as e:
        st.error(f"Application error: {e}")

if __name__ == '__main__':
    main()
