# Guardian Eye - Women Safety Analytics

## Overview

Guardian Eye is a women's safety analytics application developed for the Smart India Hackathon. The application uses computer vision and machine learning to detect emergency gestures and analyze environmental safety conditions through a camera feed, providing real-time safety alerts and recommendations.

## Features

- **Gesture Recognition**: Detects emergency gestures like help signals, SOS, and distress indicators
- **Situation Analysis**: Evaluates environmental safety based on lighting, crowd density, and surroundings
- **Risk Assessment**: Provides a comprehensive safety risk level based on multiple factors
- **Emergency Alerts**: Triggers alerts when emergency gestures or high-risk situations are detected
- **Safety Recommendations**: Offers contextual safety tips based on the current situation

## Installation

### Prerequisites

- Python 3.7 or higher
- Webcam or camera device
- Internet connection (for initial setup and model download)

### Setup

1. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Download and prepare datasets:
   ```
   python download_datasets.py
   ```

3. Train the gesture recognition model:
   ```
   python train_gesture_model.py
   ```

4. Run the application:
   ```
   python run_app.py
   ```
   
   Alternatively, you can run the Flask app directly:
   ```
   python app.py
   ```

## Usage

1. Allow camera access when prompted by your browser
2. The application will start analyzing the camera feed in real-time
3. Emergency gestures will be highlighted and alerts will be displayed
4. Safety recommendations will appear based on the current situation

### Emergency Gestures

The application recognizes several emergency gestures:

- **Help Signal**: Open palm with all five fingers extended
- **SOS Gesture**: Alternating fist and open palm
- **Pointing**: Directional pointing to indicate danger or escape route
- **Fist**: Closed fist as a distress signal

## Technical Details

### Architecture

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Flask (Python)
- **Computer Vision**: OpenCV, MediaPipe
- **Machine Learning**: scikit-learn, Random Forest Classifier

### Components

- **Gesture Recognition**: Uses MediaPipe Hands for accurate hand landmark detection and a trained Random Forest model for gesture classification
- **Situation Analysis**: Evaluates lighting conditions, crowd density, and environmental factors using computer vision techniques
- **API**: RESTful API endpoints for image processing and analysis

## Troubleshooting

### Common Issues

- **Camera Access Denied**: Ensure your browser has permission to access the camera
- **Slow Performance**: Reduce the resolution in settings or close other resource-intensive applications
- **Gesture Not Recognized**: Ensure good lighting and clear hand visibility

### Error Reporting

If you encounter any issues, check the application logs for detailed error information.