## Real-Time Person Recognition System

This Streamlit web application provides real-time person detection and recognition capabilities using YOLOv8 and OpenCV. The system supports both image uploads and live webcam feeds.

### Features

- **Multi-input Support**: Choose between image upload or webcam feed
- **Target Management**: Upload reference images of target persons
- **Real-time Detection**: Annotated video feed with bounding boxes
- **Recognition System**: Color-coded boxes for recognized targets
- **Detection Logging**: Track detected targets with timestamps
- **Visual Feedback**: Display similarity scores and confidence levels

### Enhancements

1. **Recognition Pipeline**:
   - Implemented feature embedding comparison
   - Adjustable confidence thresholds
   - Color-coding for recognized/non-recognized persons

2. **Performance Optimizations**:
   - Cached model loading
   - Batch processing of frames
   - Asynchronous I/O operations

3. **UI Improvements**:
   - Interactive sidebar controls
   - Real-time detection logging
   - Responsive layout for different devices

### Deployment

1. Create `requirements.txt` with needed dependencies
2. Set up Streamlit sharing account
3. Deploy via GitHub repository:
