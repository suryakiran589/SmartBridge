# AI-Based Hand Gesture Recognition System

A real-time AI-based Hand Gesture Recognition System that enables touchless human-computer interaction using a webcam. The system detects algorithms and classifies hand gestures using MediaPipe and OpenCV, and maps them to predefined computer actions using PyAutoGUI. It features a stunning, premium dark-mode glassmorphism interface powered by Flask.

## Features
- **Real-Time Hand Detection**: Detects and tracks multiple hands.
- **Gesture Recognition & Action Mapping**:
  - `🖐️ Open Palm` -> Mouse Movement
  - `✊ Fist` -> Scroll Down
  - `👆 Pointing` -> Volume Control (Pinch fingers to decrease, spread to increase)
  - `✌️ Peace Sign` -> Mouse Click
  - `👍 Thumbs Up` -> Play/Pause Media
- **Modern Web Dashboard**: Real-time feedback interface indicating the tracked gesture and executed action.
- **Smooth Action Execution**: Custom filtering algorithms are applied for smooth mouse control and debounce mechanisms prevent action spamming.

## Tech Stack
- **Backend**: Python 3.x, Flask, OpenCV
- **AI / Computer Vision**: Google MediaPipe Hands
- **System Control**: PyAutoGUI
- **Frontend**: HTML5, CSS3 (Glassmorphism), Vanilla JavaScript

## Project Structure
```text
hand_gesture_recognition/
│
├── app.py                     # Main Flask app server
├── requirements.txt           # Dependencies
├── README.md                  # Complete documentation
│
├── models/
│   └── gesture_model.py       # Gesture logic, heuristics, action execution
│
├── utils/
│   └── hand_tracking.py       # Hand detection & landmark extraction using MediaPipe
│
├── static/
│   ├── css/
│   │   └── style.css          # Glassmorphism UI styles
│   ├── js/
│   │   └── script.js          # Polling logic for state updates
│
├── templates/
│   └── index.html             # UI layout and video feed wrapper
│
└── assets/
    └── demo_images_or_videos/ # Optional folder for assets
```

## Quick Start (Windows)
1. Double-click the `run.bat` file located in the root folder (`ai gesture/run.bat`).
2. This automated script will:
   - Create a virtual environment (`venv`).
   - Install the required dependencies.
   - Start the Flask backend server.
3. Open your browser and navigate to `http://localhost:5002`.
4. Allow browser access to your webcam.

## Customization
If you wish to change the hardcoded heuristics or gestures, examine logic inside `hand_gesture_recognition/models/gesture_model.py`.

*Note: Since the system replicates hardware inputs (volume, mouse movement), keep a safe distance from your screen during use to avoid unintended interactions.*
