import numpy as np
import cv2
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
    """Extract eyes from an image"""
    # Convert PIL image to CV2 format
    img = pil_to_cv2(pil_image)
    
    # Load the cascade classifiers
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # List to store extracted eyes
    extracted_eyes = []
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
        # Detect eyes in the face region
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        for (ex, ey, ew, eh) in eyes:
            # Extract the eye region
            eye_img = roi_color[ey:ey+eh, ex:ex+ew]
            # Convert to PIL Image
            pil_eye = cv2_to_pil(eye_img)
            extracted_eyes.append(pil_eye)
    
    return extracted_eyes


