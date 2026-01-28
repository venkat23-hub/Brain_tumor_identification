import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image
import os
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="Brain Tumor Detection", layout="wide", initial_sidebar_state="collapsed")

# Load custom CSS
with open("static/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load model with error handling
model_path = os.path.join(os.path.dirname(__file__), "brain_tumor_model.h5")
model_url = "https://drive.google.com/uc?export=download&id=YOUR_MODEL_FILE_ID"  # Replace with actual Google Drive link

try:
    if os.path.exists(model_path):
        model = load_model(model_path)
        st.success("✅ Model loaded successfully from local file!")
    else:
        st.warning("⚠️ Local model file not found. Attempting to load from cloud storage...")
        # Uncomment and configure the line below if you want to load from a URL
        # model = load_model(model_url)
        st.error("❌ Model file not found locally. Please ensure 'brain_tumor_model.h5' is committed to your repository.")
        st.info("💡 **To fix this:**\n1. Make sure the model file is committed: `git add brain_tumor_model.h5 && git commit -m 'Add model file' && git push`\n2. Or host the model on Google Drive/Dropbox and load from URL")
        st.stop()
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.info("💡 Possible solutions:\n- Ensure the model file is in HDF5 format compatible with current Keras version\n- Check if the model was trained with a different TensorFlow version\n- Try converting the model to SavedModel format")
    st.stop()

IMG_SIZE = 128

def preprocess_image(image):
    """Preprocess the uploaded image for model prediction"""
    # Convert PIL image to numpy array
    image_array = np.array(image)
    
    # Convert to grayscale if RGB (using PIL method)
    if len(image_array.shape) == 3:
        # Convert to PIL, then to grayscale
        image = Image.fromarray(image_array.astype('uint8'))
        image = image.convert('L')
        image_array = np.array(image)
    
    # Resize to model input size using PIL
    image = Image.fromarray(image_array.astype('uint8'))
    image = image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
    image_array = np.array(image)
    
    # Normalize pixel values
    image_array = image_array.astype('float32') / 255.0
    
    # Reshape for model input (batch_size, height, width, channels)
    image_array = image_array.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    
    return image_array

# Main title
st.markdown("""
    <h1 class="main-title">Brain Tumor Detection</h1>
    <p class="subtitle">Advanced AI-powered MRI analysis for early detection</p>
""", unsafe_allow_html=True)

# Create columns for responsive layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("<h3 class='section-title'>Upload MRI Image</h3>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose an MRI image",
        type=["jpg", "png", "jpeg"],
        label_visibility="collapsed"
    )

if uploaded_file is not None:
    with col1:
        # Display uploaded image
        st.markdown("<h4 class='image-title'>Uploaded Image</h4>", unsafe_allow_html=True)
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)

    with col2:
        st.markdown("<h3 class='section-title'>Analysis Results</h3>", unsafe_allow_html=True)

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
                    <div class='result-title'>Tumor Detected</div>
                    <div class='result-message'>Abnormality Found in MRI Scan</div>
                    <div class='confidence-box'>Confidence Level: <span class='confidence-value'>{confidence*100:.2f}%</span></div>
                    <div class='warning-text'>⚠️ This result indicates potential abnormalities. Please consult a qualified medical professional for proper diagnosis and treatment.</div>
                </div>""",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""<div class='result-card no-tumor'>
                    <div class='result-title'>No Tumor Detected</div>
                    <div class='result-message'>MRI Scan Appears Normal</div>
                    <div class='confidence-box'>Confidence Level: <span class='confidence-value'>{confidence*100:.2f}%</span></div>
                    <div class='success-text'>✅ The analysis shows no signs of tumors. Continue with regular health check-ups and maintain a healthy lifestyle.</div>
                </div>""",
                unsafe_allow_html=True
            )
else:
    with col2:
        st.markdown("""
            <h3 class='info-title'>How to Use</h3>
            <ul class='info-list'>
                <li>Click the upload button to select an MRI image</li>
                <li>Supported formats: JPG, PNG, JPEG</li>
                <li>The image will be processed automatically</li>
                <li>Results will show if a tumor is detected</li>
            </ul>
            <h3 class='info-title'>Important Notes</h3>
            <ul class='info-list'>
                <li>This tool is for educational purposes</li>
                <li>Always consult a medical professional</li>
                <li>Do not rely solely on this AI for diagnosis</li>
            </ul>
        """, unsafe_allow_html=True)
