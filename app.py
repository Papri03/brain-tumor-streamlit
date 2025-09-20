# app.py
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from datetime import datetime
import pandas as pd

# ------------------------
# Load Keras CNN model
# ------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('brain_tumor_model.keras')

model = load_model()

# ------------------------
# Helper function
# ------------------------
def preprocess_image(img):
    """
    Resize and normalize image to match CNN model input shape (96,96,3)
    """
    img_resized = img.resize((96, 96))
    img_rgb = img_resized.convert('RGB')
    img_array = np.array(img_rgb) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ------------------------
# Streamlit Layout
# ------------------------
st.set_page_config(page_title="Brain Tumor Detection", layout="wide")
st.title("🧠 Brain Tumor Detection Dashboard")
st.write("Upload an MRI scan or use the demo image. The app predicts tumor type in real-time.")

# ------------------------
# Sidebar
# ------------------------
st.sidebar.title("Options")
use_demo = st.sidebar.checkbox("Use Demo MRI Image", value=False)

# Initialize session log
if 'log' not in st.session_state:
    st.session_state.log = pd.DataFrame(columns=['Timestamp', 'Filename', 'Prediction', 'Confidence'])

# Load image
image = None
if use_demo:
    try:
        image = Image.open("sample_mri.jpg")
        st.sidebar.write("Using demo MRI image")
    except FileNotFoundError:
        st.sidebar.error("Demo image not found! Please place sample_mri.jpg in the folder.")
else:
    uploaded_file = st.file_uploader("Upload your MRI image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)

# ------------------------
# Prediction
# ------------------------
if image:
    st.image(image, caption="MRI Image", use_column_width=True)
    
    try:
        img_array = preprocess_image(image)
        preds = model.predict(img_array)
        
        # Multi-class handling (softmax output)
        class_names = ['No Tumor', 'Pituitary', 'Meningioma', 'Glioma']  # replace 'Other' with correct 4th class
        pred_index = np.argmax(preds)
        pred_class = class_names[pred_index]
        confidence = preds[0][pred_index] * 100
        
        # Display results
        st.subheader("Prediction Result")
        st.write(f"**Class:** {pred_class}")
        st.progress(float(confidence) / 100)  # fixed float type for Streamlit
        st.write(f"**Confidence:** {confidence:.2f}%")
        
        # Update log
        st.session_state.log = pd.concat([st.session_state.log, pd.DataFrame([{
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Filename': "Demo Image" if use_demo else uploaded_file.name,
            'Prediction': pred_class,
            'Confidence': f"{confidence:.2f}%"
        }])], ignore_index=True)
        
    except ValueError as e:
        st.error(f"Prediction failed: {e}")

# ------------------------
# Dashboard / Logs
# ------------------------
st.subheader("📊 Prediction Log")
st.dataframe(st.session_state.log)


