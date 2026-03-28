# SignBridge — Sign Language to Speech

> Real-time sign language recognition that converts hand gestures into spoken words. Built for accessibility — no special hardware required, just a webcam.

---

## What It Does

SignBridge reads ASL/ISL hand signs through a webcam, builds them into words and sentences, and speaks them aloud. Designed for deaf and hard-of-hearing users to communicate in situations where no interpreter is available.

```
You sign letters → App builds words → Words form sentences → Sentences spoken aloud
```

---

## Tech Stack

- **MediaPipe** — hand landmark detection (21 points per hand)
- **scikit-learn** — RandomForest / MLP gesture classifier (97%+ accuracy)
- **OpenCV** — webcam capture and frame processing
- **Flask** — web server and video streaming
- **pyttsx3** — text to speech
- **PyAutoGUI** — system control fallback

---

## Project Structure

```
signbridge/
├── app.py                      # Flask server + API routes
├── train_model.py              # Dataset processing + model training
├── collect_data.py             # Custom data collection tool
├── requirements.txt            # Dependencies
│
├── models/
│   ├── gesture_model.py        # ML classifier + sentence builder + TTS
│   ├── gesture_classifier.pkl  # Trained model (generated after training)
│   ├── label_map.json          # Label map (generated after training)
│   └── hand_landmarker.task    # MediaPipe model file
│
├── utils/
│   └── hand_tracking.py        # MediaPipe hand landmark extraction
│
├── static/
│   ├── css/style.css           # UI styles
│   └── js/script.js            # Frontend polling + UI logic
│
├── templates/
│   └── index.html              # Dashboard UI
│
└── data/                       # Created during setup (gitignored)
    ├── raw/                    # Dataset images go here
    └── landmarks.csv           # Extracted landmarks (generated)
```

---

## Setup — First Time Only

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/signbridge.git
cd signbridge
```

### 2. Create virtual environment

```bash
# Mac
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing, install manually:

```bash
pip install flask opencv-python mediapipe scikit-learn pandas matplotlib seaborn joblib pyttsx3 pyautogui numpy
```

### 4. Download the dataset

1. Go to: https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset
2. Download and unzip
3. Move the `train` folder into your project:

```
data/
└── raw/
    └── asl_alphabet_train/
        ├── A/
        ├── B/
        ├── C/
        └── ...
```

### 5. Train the model

```bash
python train_model.py
```

This will:
- Extract landmarks from ~500 images per letter (~15 mins on Mac M-series, ~25 mins on Windows)
- Train RandomForest and MLP models
- Pick the better one (usually 97%+ accuracy)
- Save `models/gesture_classifier.pkl` and `models/label_map.json`
- Save a confusion matrix to `data/confusion_matrix.png`

If you already have `landmarks.csv` and just want to retrain:

```bash
python train_model.py --skip-extraction
```

### 6. Run the app

```bash
python app.py
```

Open your browser at: **http://localhost:5002**

Allow camera access when prompted.

---

## Camera Permissions

**Mac:** System Settings → Privacy & Security → Camera → enable Terminal (or VS Code / PyCharm)

**Windows:** Settings → Privacy → Camera → allow desktop apps

---

## How to Use

### Signing letters

1. Hold your hand in front of the webcam
2. Form an ASL/ISL letter sign
3. **Hold it steady** for ~1 second until the progress bar fills
4. The letter is confirmed and added to the current word

### Building words

- Sign letters one by one — they appear in the **Word** panel on the right
- **Remove your hand** for ~2 seconds to commit the word to the sentence
- The word is spoken aloud when committed

### Controls

| Button | Action |
|---|---|
| **▶ Speak** | Speak the full current sentence |
| **⌫ Delete** | Remove the last signed letter |
| **✕ Clear** | Clear everything and start over |

### Quick Phrases (one click → instant speech)

| Phrase | Use case |
|---|---|
| 👋 Hello | Greeting |
| 🆘 Help me | Emergency |
| 🙏 Thank you | Politeness |
| ✓ Yes | Confirmation |
| ✕ No | Refusal |
| 💧 Water | Basic need |
| ⚠ Pain | Medical emergency |
| ⏳ Wait | Pause conversation |

### Sentence history

The last 5 spoken sentences are saved in the history panel. Click **▶** next to any sentence to speak it again.

---

## Supported Signs

The model recognises **24 static ASL letters** (J and Z are excluded as they require motion):

```
A B C D E F G H I K L M N O P Q R S T U V W X Y
```

> **Note:** J and Z involve hand movement rather than a static pose — these can be added in a future version using sequence models (LSTM).

---

## Known Limitations

- **M and N** have lower accuracy (~80%) due to similar hand shapes — the model sometimes confuses them
- Works best in **good lighting** with the hand clearly visible against a contrasting background
- Trained on ASL — a few ISL letters differ slightly. Fine-tune with your own data using `collect_data.py` for best ISL accuracy
- **One hand only** — two-hand signs are not supported in this version

---

## Fine-tuning with Your Own Data (Optional)

Collect your own ISL samples to improve accuracy on letters that differ from ASL:

```bash
# Collect 200 samples of specific letters
python collect_data.py --person yourname --signs M N R S

# After all team members collect — merge into one dataset
python collect_data.py --person anyone --merge

# Retrain including your new data
python train_model.py --skip-extraction
```

---

## Troubleshooting

**Camera not opening on Mac:**
```bash
# Grant terminal camera permission in System Settings → Privacy → Camera
```

**`mediapipe` has no attribute `solutions`:**
```bash
pip install mediapipe==0.10.21
```

**TTS not speaking:**
```bash
# Test pyttsx3 directly
python -c "import pyttsx3; e = pyttsx3.init(); e.say('hello'); e.runAndWait()"
```

**Low accuracy / wrong predictions:**
- Make sure lighting is good — avoid backlight
- Keep hand fully visible in frame
- Sign slowly and hold steady until the progress bar fills
- Collect your own data with `collect_data.py` and retrain

**Model file not found:**
```bash
# You need to train first
python train_model.py
```

---

## Team

Built as a Final Year Project — accessible sign language communication using computer vision and machine learning.

---

## Future Work

- Dynamic gesture recognition (J, Z, and word-level signs) using LSTM
- Full ISL dataset collection and fine-tuning
- Mobile app version using MediaPipe JS
- Two-hand sign support
- Offline mode with on-device model