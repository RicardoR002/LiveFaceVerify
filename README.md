# Live Face Verification System

A Streamlit application for real-time face detection and recognition using YOLOv8.

## Features

- Face detection using YOLOv8
- Basic feature matching for person recognition
- Support for both image upload and webcam input
- Real-time detection log
- Adjustable confidence and IOU thresholds

## Deployment on Streamlit Cloud

1. Fork this repository to your GitHub account
2. Go to [Streamlit Cloud](https://share.streamlit.io/)
3. Connect your GitHub account
4. Select this repository
5. Set the main file path to `app.py`
6. Deploy!

## Local Setup

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the YOLOv8 face detection model:
   ```bash
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-face.pt
   ```
4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Upload target images in the sidebar
2. Choose between Image Upload or Webcam input
3. Adjust confidence and IOU thresholds as needed
4. View detection results and logs below

## Notes

- Webcam functionality may be limited in Streamlit Cloud
- For best results, use clear, well-lit images
- The application uses basic feature matching - for production use, consider using more sophisticated face recognition models

## Requirements

- Python 3.8+
- See requirements.txt for package dependencies
