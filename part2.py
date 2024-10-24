import cv2
import numpy as np
from typing import List, Optional, Tuple
from PIL import Image


def pil_to_cv2(pil_image):
    """Convert PIL Image to CV2 format"""
    # Convert PIL image to RGB if it's in RGBA
    if pil_image.mode == 'RGBA':
        pil_image = pil_image.convert('RGB')
    # Convert to numpy array
    numpy_image = np.array(pil_image)
    # Convert RGB to BGR
    cv2_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    return cv2_image

def cv2_to_pil(cv2_image):
    """Convert CV2 image to PIL format"""
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    # Convert to PIL Image
    pil_image = Image.fromarray(rgb_image)
    return pil_image


def return_eye(pil_image):
    max_eyes = 2
    """Extract eyes from an image"""
    # Convert PIL image to CV2 format
    img = pil_to_cv2(pil_image)
    
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        if face_cascade.empty() or eye_cascade.empty():
            raise RuntimeError("Failed to load cascade classifiers")
    except Exception as e:
        raise RuntimeError(f"Error loading cascade classifiers: {str(e)}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    # List to store eye candidates with their scores
    eye_candidates = []
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        # Detect eyes in the face region
        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20)
        )
        
        for (ex, ey, ew, eh) in eyes:
            # Calculate the relative position in the face
            relative_y = ey / h
            
            # Score the eye candidate based on:
            # 1. Vertical position (eyes should be in upper half of face)
            # 2. Size (too small or too large eyes are less likely)
            # 3. Aspect ratio (eyes should be roughly elliptical)
            position_score = 1.0 - abs(relative_y - 0.3)  # Ideal position around 30% from top
            size_score = 1.0 - abs((ew * eh) / (w * h) - 0.03)  # Ideal size about 3% of face
            aspect_ratio = ew / eh
            aspect_score = 1.0 - abs(aspect_ratio - 1.5)  # Ideal aspect ratio around 1.5
            
            total_score = position_score * size_score * aspect_score
            
            # Extract and convert the eye region
            eye_img = roi_color[ey:ey+eh, ex:ex+ew]
            pil_eye = Image.fromarray(cv2.cvtColor(eye_img, cv2.COLOR_BGR2RGB))
            
            eye_candidates.append((pil_eye, total_score))
    
    # Sort candidates by score and take the top max_eyes
    eye_candidates.sort(key=lambda x: x[1], reverse=True)
    selected_eyes = [eye for eye, _ in eye_candidates[:max_eyes]]
    
    return selected_eyes

def is_valid_eye_pair(eye1: Tuple[int, int, int, int], 
                     eye2: Tuple[int, int, int, int], 
                     face_height: int) -> bool:
    """
    Validate if two detected eyes form a likely pair based on their relative positions.
    
    Args:
        eye1: First eye coordinates (x, y, w, h)
        eye2: Second eye coordinates (x, y, w, h)
        face_height: Height of the detected face
        
    Returns:
        bool: True if the eyes form a valid pair
    """
    # Calculate centers
    center1 = (eye1[0] + eye1[2]//2, eye1[1] + eye1[3]//2)
    center2 = (eye2[0] + eye2[2]//2, eye2[1] + eye2[3]//2)
    
    # Check if eyes are roughly at the same height
    y_diff = abs(center1[1] - center2[1])
    max_y_diff = face_height * 0.1  # Allow 10% height difference
    
    # Check if eyes are horizontally separated
    x_diff = abs(center1[0] - center2[0])
    min_x_diff = face_height * 0.2  # Minimum expected eye separation
    max_x_diff = face_height * 0.6  # Maximum expected eye separation
    
    return (y_diff < max_y_diff and 
            min_x_diff < x_diff < max_x_diff)

