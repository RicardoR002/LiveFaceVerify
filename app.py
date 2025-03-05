# app.py
import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
from PIL import Image
import os
import tempfile
from datetime import datetime
import time

# Configuration
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

# Feature extraction function
def extract_features(image_path):
    try:
        # Use DeepFace to extract face embedding
        embedding = DeepFace.represent(image_path, model_name="Facenet", enforce_detection=False)
        if embedding:
            return embedding[0]['embedding']
        return None
    except Exception as e:
        st.error(f"Error extracting features: {str(e)}")
        return None

# Similarity comparison
def compare_features(input_feat, target_feat):
    if input_feat is None or target_feat is None:
        return 0
    # Calculate cosine similarity
    similarity = np.dot(input_feat, target_feat) / (
        np.linalg.norm(input_feat) * np.linalg.norm(target_feat)
    )
    st.write(f"Similarity score: {similarity:.3f}")
    return similarity

# Detection and recognition pipeline
def process_frame(frame):
    if frame is None:
        return None
    
    try:
        # Save frame temporarily for DeepFace
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            cv2.imwrite(tmp_file.name, frame)
            frame_path = tmp_file.name

        # Detect and recognize faces
        results = DeepFace.find(frame_path, db_path=TARGETS_DIR, model_name="Facenet", 
                              enforce_detection=False, silent=True)
        
        # Clean up temporary file
        os.unlink(frame_path)
        
        annotated_frame = frame.copy()
        
        if results and len(results) > 0:
            for result in results:
                if result and len(result) > 0:
                    # Get face location
                    face_data = result[0]
                    x = face_data['x']
                    y = face_data['y']
                    w = face_data['w']
                    h = face_data['h']
                    
                    # Get identity and similarity
                    identity = face_data['identity']
                    similarity = face_data['VGG-Face_cosine']
                    
                    # Draw bounding box
                    color = (0, 255, 0) if similarity > 0.7 else (0, 0, 255)
                    cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Add label
                    label = f"{os.path.splitext(os.path.basename(identity))[0]} ({similarity:.2f})"
                    cv2.putText(annotated_frame, label, (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Log detection
                    if similarity > 0.7:
                        name = os.path.splitext(os.path.basename(identity))[0]
                        if name not in st.session_state.detected_targets:
                            st.session_state.detected_targets.append({
                                "name": name,
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "confidence": similarity
                            })
        
        return annotated_frame
    except Exception as e:
        st.error(f"Error processing frame: {str(e)}")
        return None

# Sidebar controls
with st.sidebar:
    st.header("Detection Settings")
    st.write("Upload images of people you want to recognize")
    target_files = st.file_uploader("Upload Target Images", 
                                  accept_multiple_files=True,
                                  type=["jpg", "png", "jpeg"])
    
    # Process uploaded targets
    for target_file in target_files:
        try:
            # Save target image to targets directory
            target_path = os.path.join(TARGETS_DIR, target_file.name)
            with open(target_path, 'wb') as f:
                f.write(target_file.getvalue())
            
            # Extract features
            st.session_state.target_embeddings[target_file.name] = extract_features(target_path)
            
        except Exception as e:
            st.error(f"Error processing target image {target_file.name}: {str(e)}")

# Main app interface
st.title("Face Recognition System")
st.write("This application uses DeepFace with Facenet for accurate face detection and recognition.")

input_option = st.radio("Input Source", ["Image Upload", "Webcam"])

# Input handling
if input_option == "Image Upload":
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        try:
            # Create a temporary file to store the uploaded image
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                image_path = tmp_file.name
            
            # Process the image
            image = Image.open(image_path)
            frame = np.array(image)
            processed_frame = process_frame(frame)
            if processed_frame is not None:
                st.image(processed_frame, channels="BGR")
            
            # Clean up
            image.close()
            try:
                os.unlink(image_path)
            except Exception as e:
                st.warning(f"Could not delete temporary file: {str(e)}")
        except Exception as e:
            st.error(f"Error processing uploaded image: {str(e)}")

elif input_option == "Webcam":
    st.write("Note: Webcam functionality may be limited in Streamlit Cloud. Please use Image Upload for best results.")
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
                st.error("Could not open webcam. This is expected in Streamlit Cloud. Please use Image Upload instead.")
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

# Add footer with instructions
st.markdown("---")
st.markdown("""
### Instructions:
1. Upload target images in the sidebar
2. Choose between Image Upload or Webcam input
3. View detection results and logs below
""")
