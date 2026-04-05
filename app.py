from flask import Flask, render_template, Response, jsonify, request
import cv2
import platform
from models.gesture_model import GestureClassifier
import threading
import time

app = Flask(__name__)

# ── Cross-platform camera init ─────────────────────────────────────────────────
def get_camera():
    if platform.system() == "Darwin":
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap

camera     = get_camera()
classifier = GestureClassifier()
current_state = {
    "gesture": "None", "action": "Waiting...",
    "confidence": "0.00", "word": "",
    "sentence": "", "last_signed": "", "mode": "sign"
}
lock = threading.Lock()

def generate_frames():
    global current_state
    while True:
        success, frame = camera.read()
        if not success:
            break
        if platform.system() == "Darwin" and frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        processed_frame = classifier.process_frame(frame)
        with lock:
            current_state = classifier.get_current_state()
        ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/')
def dashboard():
    """Landing page / dashboard."""
    return render_template('dashboard.html')

@app.route('/app')
def app_page():
    """Main app with video feed."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/current_state')
def get_current_state():
    with lock:
        return jsonify(current_state)

@app.route('/api/speak', methods=['POST'])
def speak():
    classifier.speak_now()
    return jsonify({"status": "ok"})

@app.route('/api/speak_text', methods=['POST'])
def speak_text():
    data = request.get_json()
    text = data.get("text", "")
    if text:
        classifier.tts.speak(text)
    return jsonify({"status": "ok"})

@app.route('/api/delete', methods=['POST'])
def delete_letter():
    classifier.builder.add_sign("DELETE")
    return jsonify({"status": "ok"})

@app.route('/api/clear', methods=['POST'])
def clear():
    classifier.clear_sentence()
    return jsonify({"status": "ok"})

@app.route('/api/toggle_mode', methods=['POST'])
def toggle_mode():
    mode = classifier.toggle_mode()
    return jsonify({"status": "ok", "mode": mode})

@app.route('/api/set_mode', methods=['POST'])
def set_mode():
    """Set mode directly — used by dashboard cards."""
    data = request.get_json()
    target = data.get("mode", "sign")
    # Cycle toggle until we reach desired mode
    for _ in range(3):
        if classifier.mode == target:
            break
        classifier.toggle_mode()
    return jsonify({"status": "ok", "mode": classifier.mode})

@app.route('/api/quick_phrase', methods=['POST'])
def quick_phrase():
    data = request.get_json()
    text = data.get("text", "")
    if text:
        classifier.tts.speak(text)
    return jsonify({"status": "ok"})

# ── Run ────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n SignBridge — Sign Language to Speech")
    print(" Dashboard: http://localhost:5002")
    print(" App:       http://localhost:5002/app\n")
    app.run(host='0.0.0.0', port=5002, debug=False, threaded=True)