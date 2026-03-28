from flask import Flask, render_template, Response, jsonify
import cv2
from models.gesture_model import GestureClassifier
import threading
import time

app = Flask(__name__)

# Initialize Global Variables
camera = cv2.VideoCapture(0) # 0 for default webcam
# Reduce resolution for better performance
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

classifier = GestureClassifier()
current_state = {
    "gesture": "None",
    "action": "Waiting...",
    "confidence": "0.00"
}

# Thread lock for safe access to shared state
lock = threading.Lock()

def generate_frames():
    global current_state, classifier
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Process the frame for gestures
            processed_frame = classifier.process_frame(frame)
            
            # Update current state safely
            with lock:
                current_state = classifier.get_current_state()

            # Encode frame for streaming
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/current_state')
def get_current_state():
    """API endpoint to get the current gesture and action."""
    with lock:
        return jsonify(current_state)

if __name__ == '__main__':
    # Run the app locally over port 5002
    app.run(host='0.0.0.0', port=5002, debug=False, threaded=True)
