# app.py
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os
import tempfile
from datetime import datetime
import time

# Configuration
MODEL_PATH = "yolov8n-face.pt"
TARGETS_DIR = "targets"
os.makedirs(TARGETS_DIR, exist_ok=True)

# Initialize session state
if "detected_targets" not in st.session_state:
    st.session_state.detected_targets = []
if "target_embeddings" not in st.session_state:
    st.session_state.target_embeddings = {}
if "webcam_active" not in st.session_state:
    st.session_state.webcam_active = False
if "last_frame" not in st.session_state:
    st.session_state.last_frame = None

# Load models
@st.cache_resource
def load_model():
    try:
        return YOLO(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_model()

# Sidebar controls
with st.sidebar:
    st.header("Detection Settings")
    confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)
    iou_threshold = st.slider("IOU Threshold", 0.0, 1.0, 0.45)
    target_files = st.file_uploader("Upload Target Images", 
                                  accept_multiple_files=True,
                                  type=["jpg", "png", "jpeg"])
    
    # Process uploaded targets
    for target_file in target_files:
        try:
            target_image = Image.open(target_file)
            target_path = os.path.join(TARGETS_DIR, target_file.name)
            target_image.save(target_path)
            st.session_state.target_embeddings[target_file.name] = extract_features(target_path)
        except Exception as e:
            st.error(f"Error processing target image {target_file.name}: {str(e)}")

# Main app interface
st.title("Real-Time Person Recognition System")
input_option = st.radio("Input Source", ["Image Upload", "Webcam"])

# Feature extraction function
def extract_features(image_path):
    try:
        # For now, we'll use a simple histogram as features
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read image")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [128], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist
    except Exception as e:
        st.error(f"Error extracting features: {str(e)}")
        return None

# Similarity comparison
def compare_features(input_feat, target_feat):
    if input_feat is None or target_feat is None:
        return 0
    return np.dot(input_feat, target_feat) / (
        np.linalg.norm(input_feat) * np.linalg.norm(target_feat)
    )

# Detection and recognition pipeline
def process_frame(frame):
    if frame is None or model is None:
        return None
    
    try:
        results = model.predict(
            frame,
            conf=confidence,
            iou=iou_threshold,
            classes=[0]  # Person class
        )
        
        annotated_frame = frame.copy()
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                face_roi = frame[y1:y2, x1:x2]
                
                # Feature extraction and matching
                input_feat = extract_features(face_roi)
                best_match = None
                max_score = 0
                
                for name, target_feat in st.session_state.target_embeddings.items():
                    similarity = compare_features(input_feat, target_feat)
                    if similarity > max_score and similarity > 0.7:
                        max_score = similarity
                        best_match = name.split(".")[0]
                
                # Draw bounding box
                color = (0, 255, 0) if best_match else (0, 0, 255)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                if best_match:
                    label = f"{best_match} ({max_score:.2f})"
                    cv2.putText(annotated_frame, label, (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    if best_match not in st.session_state.detected_targets:
                        st.session_state.detected_targets.append({
                            "name": best_match,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "confidence": max_score
                        })
        
        return annotated_frame
    except Exception as e:
        st.error(f"Error processing frame: {str(e)}")
        return None

# Input handling
if input_option == "Image Upload":
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        try:
            image = Image.open(uploaded_file)
            frame = np.array(image)
            processed_frame = process_frame(frame)
            if processed_frame is not None:
                st.image(processed_frame, channels="BGR")
        except Exception as e:
            st.error(f"Error processing uploaded image: {str(e)}")

elif input_option == "Webcam":
    col1, col2 = st.columns(2)
    with col1:
        if not st.session_state.webcam_active:
            if st.button("Start Webcam"):
                st.session_state.webcam_active = True
                st.experimental_rerun()
    with col2:
        if st.session_state.webcam_active:
            if st.button("Stop Webcam"):
                st.session_state.webcam_active = False
                st.experimental_rerun()

    if st.session_state.webcam_active:
        try:
            video_capture = cv2.VideoCapture(0)
            if not video_capture.isOpened():
                st.error("Could not open webcam")
            else:
                ret, frame = video_capture.read()
                if ret:
                    processed_frame = process_frame(frame)
                    if processed_frame is not None:
                        st.image(processed_frame, channels="BGR")
                video_capture.release()
                time.sleep(0.1)  # Add small delay to prevent overwhelming the system
                st.experimental_rerun()
        except Exception as e:
            st.error(f"Error accessing webcam: {str(e)}")

# Display detection log
st.subheader("Detection Log")
if st.session_state.detected_targets:
    st.table(st.session_state.detected_targets)
else:
    st.write("No targets detected yet")