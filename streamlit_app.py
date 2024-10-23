import streamlit as st
import cv2
import numpy as np
from PIL import Image
from part1 import main  # Function for person detection
from part3 import main3  # Function for image classification

# Title of the app
st.title("🎈 My New App")
st.write(
    "Welcome to my app! Let's start building. For help and inspiration, check out the [Streamlit documentation](https://docs.streamlit.io/)."
)

# Sidebar navigation
st.sidebar.title("Navigation")
model = st.sidebar.radio("Select a model:", ("Model 1: Person Detection", "Model 2: Details", "Model 3: Image Classification"))

# Display content based on the selected model
if model == "Model 1: Person Detection":
    st.header("Model 1: Person Detection")
    
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
    st.header("Model 2")
    st.write("Details and functionality for Model 2 will go here.")
    
elif model == "Model 3: Image Classification":
    st.header("Model 3: Image Classification")
    
    uploaded_file = st.file_uploader("Upload an image:", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        # Load the uploaded image
        image = Image.open(uploaded_file)  # Use PIL to open the image

        # Button for classification
        if st.button("Classify Image"):
            result = main3(image)  # Pass the image to the main3 function
            st.write(f"🖼️ Prediction: {result}")  # Display the prediction result
