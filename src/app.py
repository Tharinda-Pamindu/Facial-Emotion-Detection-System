import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import tempfile
import os

# Settings
MODEL_PATH = 'D:/RUSL/Third Year/my/ICT3212 - Introduction to Intelligent Systems/Project/Facial Emotion Detection System/Models/emotion_model_final_v1.keras' # Or emotion_model_final.keras
# Emotion labels mapping (0=Anger, 1=Fear, 2=Happy, 3=Sad, 4=Surprise, 5=Neutral)
EMOTION_LABELS = {0: 'Anger', 1: 'Fear', 2: 'Happy', 3: 'Sad', 4: 'Surprise', 5: 'Neutral'}

# Load model
@st.cache_resource
def load_emotion_model():
    if not os.path.exists(MODEL_PATH):
        return None
    model = load_model(MODEL_PATH)
    return model

def main():
    st.title("Facial Emotion Detection System")
    st.write("Real-time emotion detection using CNN and OpenCV.")

    model = load_emotion_model()

    if model is None:
        st.error(f"Model not found at {MODEL_PATH}. Please train the model first.")
        return

    # Face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Sidebar
    st.sidebar.title("Settings")
    use_webcam = st.sidebar.checkbox("Use Webcam")

    if use_webcam:
        run_webcam(model, face_cascade)
    else:
        st.write("Check 'Use Webcam' in the sidebar to start.")

def run_webcam(model, face_cascade):
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    while True:
        _, frame = camera.read()
        if frame is None:
            break
            
        frame = cv2.flip(frame, 1) # Mirror effect
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Extract ROI
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi_gray = roi_gray.astype('float32') / 255.0
            roi_gray = np.expand_dims(roi_gray, axis=0)
            roi_gray = np.expand_dims(roi_gray, axis=-1)

            # Predict
            prediction = model.predict(roi_gray)
            max_index = int(np.argmax(prediction))
            predicted_emotion = EMOTION_LABELS[max_index]
            confidence = prediction[0][max_index]

            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{predicted_emotion} ({confidence:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        FRAME_WINDOW.image(frame, channels='BGR')

    camera.release()

if __name__ == '__main__':
    main()
