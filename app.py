import streamlit as st
import cv2
from deepface import DeepFace
import numpy as np

def main():
    st.title("Face Emotion Recognition App")

    # Sidebar elements
    st.sidebar.title("Settings")
    detection_confidence = st.sidebar.slider("Detection Confidence", 0.0, 1.0, 0.5, 0.1)
    show_bbox = st.sidebar.checkbox("Show Bounding Box", True)
    show_confidence = st.sidebar.checkbox("Show Confidence", True)

    # Main app
    img_file_buffer = st.camera_input("Take a picture")

    if img_file_buffer is not None:
        # Read image from buffer
        bytes_data = img_file_buffer.getvalue()
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        # Perform emotion analysis
        results = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)

        # Display results
        for face in results:
            bbox = face['region']
            emotion = face['dominant_emotion']
            emotion_confidence = face['emotion'][emotion]

            if show_bbox:
                cv2.rectangle(img, (bbox['x'], bbox['y']), 
                              (bbox['x']+bbox['w'], bbox['y']+bbox['h']), 
                              (0, 255, 0), 2)

            label = f"{emotion}"
            if show_confidence:
                label += f": {emotion_confidence:.2f}"

            cv2.putText(img, label, (bbox['x'], bbox['y']-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        st.image(img, channels="BGR")

if __name__ == "__main__":
    main()
