import streamlit as st
import cv2
import numpy as np
from PIL import Image
from part1 import main  # Function for person detection
from part3 import main3  # Function for image classification

# Title of the app
st.title("🎈 CSC-425-3-STEP-MODEL-DETECTION 🎈")
st.write(
    "This app here is to test the machine learning model with proper GUI!"
)

# Sidebar navigation
st.sidebar.title("Navigation")
model = st.sidebar.radio("Select a model:", ("Model 1: Person Detection", "Model 2: Details", "Model 3: Image Classification"))

# Display content based on the selected model
if model == "Model 1: Person Detection":
    st.header("Model 1: Person Detection")
    st.subheader("Detect whether an image has a driver or not!")
    
    uploaded_file = st.file_uploader("Upload an image:", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        # Read the uploaded image
        image_bytes = uploaded_file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Button for detection
        if st.button("Detect Person"):
            person_detected, score = main(image)  # Pass the image to the main function
            st.write(f"👤 Person detected: {person_detected}")  # Display result
    
elif model == "Model 2: Details":
    st.header("Model 2: Eye Extraction")
    st.subheader("Extract the eye from the image of the driver.")
    
elif model == "Model 3: Image Classification":
    st.header("Model 3: Sleepiness Detection!")
    st.subheader("Check for drowsiness in the eye of the driver.")
    uploaded_file = st.file_uploader("Upload an image:", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        # Load the uploaded image
        image = Image.open(uploaded_file)  # Use PIL to open the image

        # Button for classification
        if st.button("Classify Image"):
            result = main3(image)  # Pass the image to the main3 function
            st.write(f"🖼️ Prediction: {result}")  # Display the prediction result
