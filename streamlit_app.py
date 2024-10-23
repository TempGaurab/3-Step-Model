import streamlit as st
import cv2
import numpy as np
from PIL import Image
from part1 import main  # Function for person detection
from part3 import main3  # Function for image classification
from part2 import return_eye
from io import BytesIO
import os
import base64

def get_image_download_bytes(pil_image, format='PNG'):
    """Convert PIL Image to bytes for downloading"""
    buf = BytesIO()
    pil_image.save(buf, format=format)
    return buf.getvalue()

def cv2_to_pil(cv2_image):
    """Convert CV2 image to PIL format"""
    cv2_image_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cv2_image_rgb)
    return pil_image

# Title of the app
st.title("🎈 CSC-425-3-STEP-MODEL-DETECTION 🎈")
st.write(
    '''
The proposed system uses a three-model approach for driver drowsiness detection, where the first model activates only when the car is running and checks for a person in the driver's seat. When a person is detected, the second model locates the driver's eyes and sends these images to a custom-built third model. The third model analyzes eye features to detect drowsiness, potentially triggering various alert mechanisms like intermittent braking or honking if the driver appears to be sleeping. 
    '''
)

# Sidebar navigation
st.sidebar.title("Navigation")
model = st.sidebar.radio("Select a model:", 
    ("Model 1: Person Detection", 
     "Model 2: Eye Extraction", 
     "Model 3: Image Classification",
     "Models Together"))

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
        
        # Display uploaded image
        pil_image = cv2_to_pil(image)
        st.image(pil_image, caption='Uploaded Image', use_column_width=True)
        
        # Button for detection
        if st.button("Detect Person"):
            person_detected, score = main(image)  # Pass the image to the main function
            st.write(f"👤 Person detected: {person_detected}")  # Display result
            
            # Add download button for the processed image
            img_bytes = get_image_download_bytes(pil_image)
            st.download_button(
                label="Download Image",
                data=img_bytes,
                file_name="person_detection.png",
                mime="image/png"
            )
    else:
        st.warning("Please upload an image to detect a person.")

elif model == "Model 2: Eye Extraction":
    st.header("Model 2: Eye Extraction")
    st.subheader("Extract the eye from the image of the driver.")

    uploaded_file = st.file_uploader("Upload an image:", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        if st.button("Extract Eyes"):
            with st.spinner("Extracting eyes..."):
                try:
                    eye_images = return_eye(image)
                    if eye_images:
                        st.success(f"Found {len(eye_images)} eyes!")
                        
                        # Display extracted eyes
                        for i, eye_img in enumerate(eye_images):
                            st.image(eye_img, caption=f'Extracted Eye {i+1}', use_column_width=True)
                        
                        # Create ZIP file with all eyes
                        zip_buffer = BytesIO()
                        import zipfile
                        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                            for i, eye_img in enumerate(eye_images):
                                img_bytes = get_image_download_bytes(eye_img)
                                zip_file.writestr(f"extracted_eye_{i+1}.png", img_bytes)
                        
                        # Add download button
                        st.download_button(
                            label="Download Extracted Eyes",
                            data=zip_buffer.getvalue(),
                            file_name="extracted_eyes.zip",
                            mime="application/zip"
                        )
                    else:
                        st.warning("No eyes detected in the image.")
                except Exception as e:
                    st.error(f"Error during eye extraction: {str(e)}")
    else:
        st.warning("Please upload an image to extract eyes.")

elif model == "Model 3: Image Classification":
    st.header("Model 3: Sleepiness Detection!")
    st.subheader("Check for drowsiness in the eye of the driver.")
    
    uploaded_file = st.file_uploader("Upload an image:", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        # Load the uploaded image
        image = Image.open(uploaded_file)
        
        # Display uploaded image
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Button for classification
        if st.button("Classify Image"):
            result = main3(image)  # Pass the image to the main3 function
            st.write(f"🖼️ Prediction: {result}")  # Display the prediction result
            
            # Add download button for the processed image
            img_bytes = get_image_download_bytes(image)
            st.download_button(
                label="Download Image",
                data=img_bytes,
                file_name="drowsiness_detection.png",
                mime="image/png"
            )
    else:
        st.warning("Please upload an image for drowsiness detection.")

elif model == "Models Together":
    st.header("Complete Drowsiness Detection Pipeline")
    st.subheader("Run all three models in sequence")
    uploaded_file = st.file_uploader("Upload an image:", type=["png", "jpg", "jpeg"], key="combined")
    
    if uploaded_file is not None:
        # Read and display the uploaded image
        image_bytes = uploaded_file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        cv2_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        pil_image = cv2_to_pil(cv2_image)
        st.image(pil_image, caption='Uploaded Image', use_column_width=True)
        
        if st.button("Run Complete Analysis"):
            st.write("🔄 Running complete analysis...")
            
            # Step 1: Person Detection
            st.write("Step 1: Person Detection")
            person_detected, score = main(cv2_image)
            st.write(f"👤 Person detected: {person_detected}")
            
            if person_detected:
                # Step 2: Eye Extraction
                st.write("Step 2: Eye Extraction")
                try:
                    eye_images = return_eye(pil_image)
                    if eye_images:
                        st.success(f"Found {len(eye_images)} eyes!")
                        
                        # Display and analyze each eye
                        for i, eye_img in enumerate(eye_images):
                            st.image(eye_img, caption=f'Extracted Eye {i+1}', use_column_width=True)
                            
                            # Step 3: Drowsiness Detection for each eye
                            st.write(f"Step 3: Drowsiness Detection for Eye {i+1}")
                            drowsiness_result = main3(eye_img)
                            st.write(f"🖼️ Prediction for Eye {i+1}: {drowsiness_result}")
                        
                        # Create ZIP file with all processed images
                        zip_buffer = BytesIO()
                        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                            # Save original image
                            orig_bytes = get_image_download_bytes(pil_image)
                            zip_file.writestr("original_image.png", orig_bytes)
                            
                            # Save extracted eyes
                            for i, eye_img in enumerate(eye_images):
                                img_bytes = get_image_download_bytes(eye_img)
                                zip_file.writestr(f"extracted_eye_{i+1}.png", img_bytes)
                        
                        # Add download button for all images
                        st.download_button(
                            label="Download All Images",
                            data=zip_buffer.getvalue(),
                            file_name="complete_analysis.zip",
                            mime="application/zip"
                        )
                    else:
                        st.warning("No eyes detected in the image.")
                except Exception as e:
                    st.error(f"Error during eye extraction: {str(e)}")
            else:
                st.warning("No person detected in the image. Stopping analysis.")
    else:
        st.warning("Please upload an image to begin the complete analysis.")