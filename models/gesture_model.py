"""
gesture_model.py  —  ML-powered ISL/ASL Sign Language Classifier
=================================================================
Drop-in replacement for the original heuristic-based GestureClassifier.
app.py requires ZERO changes — same class name, same methods.

New capabilities:
  - ML model (RandomForest or MLP) replaces finger-counting heuristics
  - Recognises A-Z (minus J,Z) + 0-9 = 34 sign classes
  - Hold-to-confirm: sign must be stable for N frames before accepted
  - Sentence builder: signs → words → sentences
  - Text-to-speech: speaks completed sentences aloud
  - Confidence threshold: rejects uncertain predictions
"""

import cv2
import numpy as np
import json
import os
import platform
import threading
import joblib
import pyautogui
from utils.hand_tracking import HandTracker

# ── Config ──────────────────────────────────────────────────────────────────────

MODEL_PATH     = os.path.join(os.path.dirname(__file__), "gesture_classifier.pkl")
LABEL_MAP_PATH = os.path.join(os.path.dirname(__file__), "label_map.json")

CONFIDENCE_THRESHOLD = 0.75   # below this → rejected as "None"
HOLD_FRAMES          = 20     # frames sign must be stable before accepted
HOLD_THRESHOLD       = 0.12   # max feature drift allowed during hold
COOLDOWN_FRAMES      = 30     # frames to wait after accepting a sign

# UI colors (BGR)
COLOR_GREEN  = (0, 220, 80)
COLOR_AMBER  = (0, 165, 255)
COLOR_RED    = (0, 60, 220)
COLOR_WHITE  = (255, 255, 255)
COLOR_GRAY   = (160, 160, 160)
COLOR_PURPLE = (200, 80, 200)


# ── TTS engine (runs in background thread to avoid blocking video) ──────────────

class TTSEngine:
    def __init__(self):
        self._engine = None
        self._lock   = threading.Lock()
        self._init()

    def _init(self):
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", 150)   # words per minute
            self._engine.setProperty("volume", 1.0)
        except Exception as e:
            print(f"[TTS] Warning: pyttsx3 not available ({e}). Install with: pip install pyttsx3")
            self._engine = None

    def speak(self, text):
        """Speaks text in a background thread — non-blocking."""
        if not self._engine or not text.strip():
            return
        def _run():
            with self._lock:
                try:
                    self._engine.say(text)
                    self._engine.runAndWait()
                except Exception:
                    pass
        threading.Thread(target=_run, daemon=True).start()


# ── Sentence builder ────────────────────────────────────────────────────────────

class SentenceBuilder:
    """
    Accumulates confirmed signs into letters → words → sentences.
    Special signs:
        SPACE   → commit current word to sentence
        DELETE  → remove last letter
        CLEAR   → clear entire sentence
    """
    def __init__(self):
        self.current_word : list[str] = []
        self.sentence     : list[str] = []

    def add_sign(self, sign: str):
        sign = sign.upper()
        if sign == "SPACE":
            if self.current_word:
                self.sentence.append("".join(self.current_word))
                self.current_word = []
        elif sign == "DELETE":
            if self.current_word:
                self.current_word.pop()
            elif self.sentence:
                # restore last word for editing
                self.current_word = list(self.sentence.pop())
        elif sign == "CLEAR":
            self.current_word = []
            self.sentence     = []
        else:
            self.current_word.append(sign)

    @property
    def current_word_str(self) -> str:
        return "".join(self.current_word)

    @property
    def sentence_str(self) -> str:
        words = self.sentence + ([self.current_word_str] if self.current_word else [])
        return " ".join(words)

    @property
    def completed_sentence(self) -> str:
        """Returns only the committed words (not the word being typed)."""
        return " ".join(self.sentence)


# ── Feature extraction (must match train_model.py exactly) ──────────────────────

def extract_features(lm_list):
    """
    Takes find_position() output: [[id, cx, cy], ...]
    Returns normalized 63-dim feature vector or None if invalid.
    """
    if len(lm_list) != 21:
        return None

    coords = [(pt[1], pt[2], 0.0) for pt in lm_list]  # x, y, z(=0 from pixel coords)
    wx, wy, _ = coords[0]

    xs    = [c[0] for c in coords]
    ys    = [c[1] for c in coords]
    scale = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)

    features = []
    for (x, y, z) in coords:
        features.extend([(x - wx) / scale, (y - wy) / scale, z])
    return features


# ── Main classifier ─────────────────────────────────────────────────────────────

class GestureClassifier:
    """
    Drop-in replacement for the original heuristic GestureClassifier.
    Identical public interface: process_frame(), get_current_state()
    """

    def __init__(self):
        self.tracker = HandTracker(detection_con=0.8, track_con=0.7)

        # Load trained model
        self._model    = None
        self._classes  = []
        self._load_model()

        # Sentence builder + TTS
        self.builder = SentenceBuilder()
        self.tts     = TTSEngine()

        # State
        self.active_gesture  = "None"
        self.active_action   = "Waiting..."
        self.confidence      = 0.0
        self.last_signed     = ""

        # Hold-to-confirm state
        self._hold_sign      = None
        self._hold_count     = 0
        self._last_features  = None
        self._cooldown       = 0

        # Keep PyAutoGUI for mouse/volume (backwards compat with original gestures)
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE    = 0.01
        self.screen_w, self.screen_h = pyautogui.size()
        self.cam_w, self.cam_h       = 640, 480
        self.ploc_x = self.ploc_y    = 0
        self.smoothing               = 5
        self.frame_reduction         = 80

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self):
        if not os.path.exists(MODEL_PATH):
            print(f"[GestureClassifier] WARNING: Model not found at {MODEL_PATH}")
            print("  Run:  python train_model.py")
            print("  Falling back to heuristic mode.\n")
            return

        try:
            bundle          = joblib.load(MODEL_PATH)
            self._model     = bundle["model"]
            self._classes   = bundle["classes"]
            model_name      = bundle.get("model_name", "unknown")
            accuracy        = bundle.get("accuracy", 0)
            print(f"[GestureClassifier] Model loaded: {model_name} ({accuracy*100:.1f}% accuracy)")
            print(f"  Classes ({len(self._classes)}): {self._classes}")
        except Exception as e:
            print(f"[GestureClassifier] ERROR loading model: {e}")

    # ── ML prediction ─────────────────────────────────────────────────────────

    def _predict(self, lm_list):
        """Returns (label, confidence) or ('None', 0.0) if model not loaded."""
        if self._model is None:
            return "None", 0.0

        features = extract_features(lm_list)
        if features is None:
            return "None", 0.0

        X      = np.array(features).reshape(1, -1)
        probs  = self._model.predict_proba(X)[0]
        idx    = probs.argmax()
        conf   = probs[idx]

        if conf < CONFIDENCE_THRESHOLD:
            return "None", float(conf)

        return self._classes[idx], float(conf)

    # ── Hold-to-confirm logic ─────────────────────────────────────────────────

    def _update_hold(self, sign, features):
        """
        Returns the confirmed sign once it has been held stable for
        HOLD_FRAMES consecutive frames. Returns None otherwise.
        """
        if sign == "None" or features is None:
            self._hold_count = max(0, self._hold_count - 2)
            self._hold_sign  = None
            return None

        # Check if same sign as last frame
        if sign != self._hold_sign:
            self._hold_sign  = sign
            self._hold_count = 1
            self._last_features = features
            return None

        # Check positional stability
        if self._last_features is not None:
            drift = np.linalg.norm(
                np.array(features) - np.array(self._last_features)
            )
            if drift > HOLD_THRESHOLD:
                self._hold_count = max(0, self._hold_count - 1)
                self._last_features = features
                return None

        self._hold_count   += 1
        self._last_features = features

        if self._hold_count >= HOLD_FRAMES:
            self._hold_count = 0
            self._hold_sign  = None
            return sign

        return None

    # ── Frame processing (public API — called by app.py) ──────────────────────

    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        frame = self.tracker.find_hands(frame)
        lm_list = self.tracker.find_position(frame, draw=False)

        self.active_gesture = "None"
        self.active_action  = "Show a sign..."
        self.confidence     = 0.0

        if self._cooldown > 0:
            self._cooldown -= 1

        if len(lm_list) == 21:
            # If model is loaded — use ML classification
            if self._model is not None:
                sign, conf  = self._predict(lm_list)
                features    = extract_features(lm_list)
                self.confidence     = conf
                self.active_gesture = sign if sign != "None" else "..."

                # Hold-to-confirm
                confirmed = None
                if self._cooldown == 0:
                    confirmed = self._update_hold(sign, features)

                if confirmed:
                    self.builder.add_sign(confirmed)
                    self.last_signed   = confirmed
                    self.active_action = f"Signed: {confirmed}"
                    self._cooldown     = COOLDOWN_FRAMES

                    # Speak completed sentence when SPACE is signed
                    if confirmed == "SPACE" and self.builder.completed_sentence:
                        self.tts.speak(self.builder.completed_sentence)
                        self.active_action = f"Speaking: {self.builder.completed_sentence}"
                else:
                    # Show hold progress
                    if self._hold_count > 0 and sign != "None":
                        pct = int(self._hold_count / HOLD_FRAMES * 100)
                        self.active_action = f"Hold... {pct}%"
                    else:
                        self.active_action = "Reading sign..."

            else:
                # Fallback: original heuristic gestures if model not loaded
                frame = self._heuristic_fallback(frame, lm_list)

        # Draw UI overlay
        frame = self._draw_ui(frame)
        return frame

    # ── UI drawing ────────────────────────────────────────────────────────────

    def _draw_ui(self, frame):
        h, w = frame.shape[:2]

        # Semi-transparent top bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (20, 20, 20), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # Current gesture (large)
        conf_color = COLOR_GREEN if self.confidence >= CONFIDENCE_THRESHOLD else COLOR_AMBER
        cv2.putText(frame, self.active_gesture, (16, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, conf_color, 2)

        # Confidence bar
        bar_w = int(180 * min(self.confidence, 1.0))
        cv2.rectangle(frame, (16, 58), (196, 70), (60, 60, 60), -1)
        cv2.rectangle(frame, (16, 58), (16 + bar_w, 70), conf_color, -1)
        cv2.putText(frame, f"{self.confidence*100:.0f}%", (202, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_GRAY, 1)

        # Action status
        cv2.putText(frame, self.active_action, (16, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 1)

        # Hold progress bar (when stabilising)
        if self._hold_count > 0:
            hold_w = int((w - 32) * self._hold_count / HOLD_FRAMES)
            cv2.rectangle(frame, (16, h - 24), (w - 16, h - 14), (60, 60, 60), -1)
            cv2.rectangle(frame, (16, h - 24), (16 + hold_w, h - 14), COLOR_GREEN, -1)
            cv2.putText(frame, "Hold steady", (16, h - 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_GRAY, 1)

        # Sentence builder overlay (bottom)
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, h - 90), (w, h - 26), (20, 20, 20), -1)
        frame = cv2.addWeighted(overlay2, 0.55, frame, 0.45, 0)

        word_display = f"Word: {self.builder.current_word_str}_" if self.builder.current_word_str else "Word: _"
        sent_display = f"Sentence: {self.builder.sentence_str}" if self.builder.sentence_str else "Sentence: (sign letters, SPACE=word break)"

        cv2.putText(frame, word_display, (16, h - 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_AMBER, 1)
        cv2.putText(frame, sent_display[:72], (16, h - 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)

        return frame

    # ── Heuristic fallback (original logic — used if model not loaded) ─────────

    def _heuristic_fallback(self, frame, lm_list):
        tip_ids = [4, 8, 12, 16, 20]
        fingers = []
        if lm_list[tip_ids[0]][1] > lm_list[tip_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        for id in range(1, 5):
            fingers.append(1 if lm_list[tip_ids[id]][2] < lm_list[tip_ids[id] - 2][2] else 0)

        total = fingers.count(1)
        self.confidence = 0.95

        if total == 5:
            self.active_gesture = "Open Palm"
            self.active_action  = "Mouse Movement"
            x1, y1 = lm_list[8][1], lm_list[8][2]
            x3 = np.interp(x1, (self.frame_reduction, self.cam_w - self.frame_reduction), (0, self.screen_w))
            y3 = np.interp(y1, (self.frame_reduction, self.cam_h - self.frame_reduction), (0, self.screen_h))
            self.cloc_x = self.ploc_x + (x3 - self.ploc_x) / self.smoothing
            self.cloc_y = self.ploc_y + (y3 - self.ploc_y) / self.smoothing
            try:
                pyautogui.moveTo(self.screen_w - self.cloc_x, self.cloc_y)
                self.ploc_x, self.ploc_y = self.cloc_x, self.cloc_y
            except Exception:
                pass
        elif total == 0:
            self.active_gesture = "Fist"
            self.active_action  = "Scroll Down"
            pyautogui.scroll(-300)

        return frame

    # ── Public API ────────────────────────────────────────────────────────────

    def get_current_state(self):
        """Called by app.py — identical return format as original."""
        return {
            "gesture":    self.active_gesture,
            "action":     self.active_action,
            "confidence": f"{self.confidence:.2f}",
            "word":       self.builder.current_word_str,
            "sentence":   self.builder.sentence_str,
            "last_signed": self.last_signed
        }

    def clear_sentence(self):
        """Called if you add a 'clear' button to the frontend."""
        self.builder.add_sign("CLEAR")

    def speak_now(self):
        """Manually trigger TTS for current sentence."""
        text = self.builder.sentence_str
        if text:
            self.tts.speak(text)