import numpy as np
import cv2

def get_eyes(faces, eye_cascade, gray, img):
    # Initialize a list to store the extracted eye images
    eye_images = []

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        # Detect eyes in the face region
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            eye_img = roi_color[ey:ey+eh, ex:ex+ew]  # Extract the eye region

            # Append the extracted eye image to the list
            eye_images.append(eye_img)

    return eye_images  # Return the list of extracted eye images


def return_eye(img):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # Initialize an eye counter for naming the saved images
    get_eyes(faces,eye_cascade,gray,img)




