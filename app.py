import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from PIL import Image
import os

# Page config
st.set_page_config(page_title="Brain Tumor Detection", layout="wide", initial_sidebar_state="collapsed")

# Load custom CSS
with open("static/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load model
model_path = os.path.join(os.path.dirname(__file__), "brain_tumor_model.h5")
model = load_model(model_path)

IMG_SIZE = 128

def preprocess_image(image):
    """Preprocess the uploaded image for model prediction"""
    # Convert PIL image to numpy array
    image_array = np.array(image)
    
    # Convert to grayscale if RGB
    if len(image_array.shape) == 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    
    # Resize to model input size
    image_array = cv2.resize(image_array, (IMG_SIZE, IMG_SIZE))
    
    # Normalize pixel values
    image_array = image_array.astype('float32') / 255.0
    
    # Reshape for model input (batch_size, height, width, channels)
    image_array = image_array.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    
    return image_array

# Main title
st.markdown("""
    <div class="header-container">
        <h1 class="main-title">🧠 Brain Tumor Detection</h1>
        <p class="subtitle">Upload an MRI image to detect brain tumors</p>
    </div>
""", unsafe_allow_html=True)

# Create columns for responsive layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
    st.markdown("<h3 class='section-title'>📤 Upload MRI Image</h3>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an MRI image",
        type=["jpg", "png", "jpeg"],
        label_visibility="collapsed"
    )
    st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is not None:
    with col1:
        # Display uploaded image
        st.markdown("<h4 class='image-title'>Uploaded Image</h4>", unsafe_allow_html=True)
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
    
    with col2:
        st.markdown("<div class='result-section'>", unsafe_allow_html=True)
        st.markdown("<h3 class='section-title'>🔍 Analysis Results</h3>", unsafe_allow_html=True)
        
        # Preprocess image
        processed = preprocess_image(image)
        
        # Make prediction
        with st.spinner("Analyzing image..."):
            prediction = model.predict(processed, verbose=0)
        
        confidence = float(np.max(prediction))
        result = int(np.argmax(prediction))
        
        # Display results
        if result == 1:
            st.markdown(
                f"""<div class='result-card tumor-detected'>
                    <div class='result-icon'>⚠️</div>
                    <div class='result-title'>Tumor Detected</div>
                    <div class='result-message'>Abnormality Found</div>
                    <div class='confidence-box'>Confidence: <span class='confidence-value'>{confidence*100:.2f}%</span></div>
                    <div class='warning-text'>Please consult a medical professional for further evaluation.</div>
                </div>""",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""<div class='result-card no-tumor'>
                    <div class='result-icon'>✅</div>
                    <div class='result-title'>You are Safe!!</div>
                    <div class='result-message'>No Tumor Detected</div>
                    <div class='confidence-box'>Confidence: <span class='confidence-value'>{confidence*100:.2f}%</span></div>
                    <div class='success-text'>The scan appears to be normal. Continue with regular check-ups.</div>
                </div>""",
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)
else:
    with col2:
        st.markdown("""
            <div class='info-section'>
                <h3 class='info-title'>ℹ️ How to Use</h3>
                <ul class='info-list'>
                    <li>Click the upload button to select an MRI image</li>
                    <li>Supported formats: JPG, PNG, JPEG</li>
                    <li>The image will be processed automatically</li>
                    <li>Results will show if a tumor is detected</li>
                </ul>
                <h3 class='info-title'>⚕️ Important Notes</h3>
                <ul class='info-list'>
                    <li>This tool is for educational purposes</li>
                    <li>Always consult a medical professional</li>
                    <li>Do not rely solely on this AI for diagnosis</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
